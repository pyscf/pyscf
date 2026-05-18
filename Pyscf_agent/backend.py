#!/usr/bin/env python

'''
Shared backend for the maintained PySCF agent module.

The backend keeps the existing workflow nodes and adds:
1. A unified request entrypoint for CLI and web frontends
2. Structured messages attached to the workflow state
3. Structured logs for each major execution step
4. Optional LangGraph graph construction when langgraph is installed
'''

from __future__ import annotations

import copy
import inspect
import io
import json
import logging
import re
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_METHODS = ('hf', 'dft')
SUPPORTED_JOBS = ('single_point',)
DEFAULT_ANALYSIS = (
    'energy',
    'homo_lumo',
    'mulliken',
    'dipole',
)
REQUEST_HINT_KEYS = (
    'request',
    'message',
    'prompt',
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MessageEnvelope:
    role: str
    kind: str
    content: str
    channel: str = 'agent'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class LogEntry:
    level: str
    event: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_timestamp)


@dataclass
class SystemSpec:
    atom: Optional[str] = None
    basis: Optional[str] = None
    unit: str = 'Angstrom'
    charge: int = 0
    spin: int = 0
    symmetry: bool = False


@dataclass
class MethodSpec:
    name: str = 'hf'
    restricted: Optional[bool] = None
    xc: Optional[str] = None


@dataclass
class JobSpec:
    name: str = 'single_point'


@dataclass
class AnalysisSpec:
    outputs: List[str] = field(default_factory=lambda: list(DEFAULT_ANALYSIS))


@dataclass
class RuntimeSpec:
    max_cycle: int = 50
    conv_tol: Optional[float] = None
    verbose: int = 4


@dataclass
class TaskSpec:
    system: SystemSpec = field(default_factory=SystemSpec)
    method: MethodSpec = field(default_factory=MethodSpec)
    job: JobSpec = field(default_factory=JobSpec)
    analysis: AnalysisSpec = field(default_factory=AnalysisSpec)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)


def message_to_dict(message: MessageEnvelope) -> Dict[str, Any]:
    return asdict(message)


def log_entry_to_dict(entry: LogEntry) -> Dict[str, Any]:
    return asdict(entry)


def normalize_message(
    user_request: Any,
    *,
    channel: str = 'agent',
    role: str = 'user',
    kind: str = 'request',
) -> Dict[str, Any]:
    if isinstance(user_request, dict):
        metadata = copy.deepcopy(user_request.get('metadata') or {})
        return message_to_dict(MessageEnvelope(
            role=user_request.get('role', role),
            kind=user_request.get('kind', kind),
            content=user_request.get('content', ''),
            channel=user_request.get('channel', channel),
            metadata=metadata,
        ))
    return message_to_dict(MessageEnvelope(
        role=role,
        kind=kind,
        content=str(user_request),
        channel=channel,
    ))


def append_message(
    state: Dict[str, Any],
    *,
    role: str,
    kind: str,
    content: str,
    channel: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    state.setdefault('messages', []).append(message_to_dict(MessageEnvelope(
        role=role,
        kind=kind,
        content=content,
        channel=channel or state.get('channel', 'agent'),
        metadata=copy.deepcopy(metadata or {}),
    )))
    return state


def append_log(
    state: Dict[str, Any],
    level: str,
    event: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry = log_entry_to_dict(LogEntry(
        level=level,
        event=event,
        details=copy.deepcopy(details or {}),
    ))
    state.setdefault('logs', []).append(entry)
    log_level = getattr(logging, level.upper(), logging.INFO)
    LOGGER.log(log_level, json.dumps(entry, ensure_ascii=False, sort_keys=True))
    return state


def default_state(user_request: Any, *, channel: str = 'agent') -> Dict[str, Any]:
    message = normalize_message(user_request, channel=channel)
    state = {
        'channel': message['channel'],
        'user_request': message['content'],
        'task_spec': None,
        'validation_errors': [],
        'clarification_questions': [],
        'generated_input': None,
        'execution_status': 'pending',
        'raw_stdout': '',
        'raw_stderr': '',
        'structured_results': None,
        'analysis_summary': '',
        'retry_count': 0,
        'max_retries': 1,
        'messages': [message],
        'logs': [],
    }
    append_log(state, 'info', 'workflow.initialized', {
        'channel': state['channel'],
        'kind': message['kind'],
    })
    return state


def task_spec_to_dict(task_spec: TaskSpec) -> Dict[str, Any]:
    return asdict(task_spec)


def task_spec_from_dict(data: Optional[Dict[str, Any]]) -> TaskSpec:
    if data is None:
        return TaskSpec()
    return TaskSpec(
        system=SystemSpec(**data.get('system', {})),
        method=MethodSpec(**data.get('method', {})),
        job=JobSpec(**data.get('job', {})),
        analysis=AnalysisSpec(**data.get('analysis', {})),
        runtime=RuntimeSpec(**data.get('runtime', {})),
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    '''Return the first valid JSON object embedded in free-form text.'''
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:idx + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _extract_key_value_block(text: str) -> Dict[str, Any]:
    '''Parse simple ``key: value`` lines into a flat task description.'''
    parsed = {}
    for line in text.splitlines():
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        key = key.strip().lower()
        value = value.strip()
        if not value:
            continue
        if key in ('atom', 'basis', 'method', 'job', 'xc', 'unit'):
            parsed[key] = value
        elif key in ('charge', 'spin', 'max_cycle', 'verbose'):
            parsed[key] = int(value)
        elif key == 'symmetry':
            parsed[key] = value.lower() in ('1', 'true', 'yes', 'on')
        elif key == 'conv_tol':
            parsed[key] = float(value)
        elif key == 'outputs':
            parsed[key] = [item.strip() for item in value.split(',') if item.strip()]
    return parsed


def parse_user_request(user_request: str) -> Dict[str, Any]:
    parsed = _extract_json_object(user_request)
    request_hints = []
    if parsed is None:
        parsed = _extract_key_value_block(user_request)
    else:
        for key in REQUEST_HINT_KEYS:
            value = parsed.get(key)
            if isinstance(value, str):
                normalized_value = value.strip()
                if normalized_value and normalized_value not in request_hints:
                    request_hints.append(normalized_value)
    lower_source = request_hints if parsed is not None else [user_request]
    lower = '\n'.join(lower_source).lower()

    basis_match = re.search(r'([a-z0-9+\-]+g(?:\*{1,2})?)', lower)
    if 'basis' not in parsed and basis_match:
        parsed['basis'] = basis_match.group(1)

    if 'method' not in parsed:
        if 'dft' in lower or 'b3lyp' in lower or 'pbe' in lower:
            parsed['method'] = 'dft'
        elif 'hf' in lower or 'hartree-fock' in lower:
            parsed['method'] = 'hf'

    if 'xc' not in parsed:
        for xc_name in ('b3lyp', 'pbe0', 'pbe', 'lda'):
            if xc_name in lower:
                parsed['xc'] = xc_name
                break

    if 'job' not in parsed:
        parsed['job'] = 'single_point'

    if 'outputs' not in parsed:
        outputs = []
        if 'homo' in lower or 'lumo' in lower:
            outputs.append('homo_lumo')
        if 'dipole' in lower:
            outputs.append('dipole')
        if 'mulliken' in lower or 'population' in lower:
            outputs.append('mulliken')
        if 'energy' in lower or 'single point' in lower or 'single-point' in lower:
            outputs.append('energy')
        if outputs:
            parsed['outputs'] = sorted(set(outputs))

    return parsed


def intent_parser(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    parsed = parse_user_request(state['user_request'])
    if state.get('task_spec') is None:
        state['task_spec'] = parsed
    else:
        merged = copy.deepcopy(state['task_spec'])
        merged.update(parsed)
        state['task_spec'] = merged
    append_log(state, 'info', 'workflow.intent_parsed', {
        'keys': sorted(state['task_spec'].keys()),
    })
    append_message(state, role='system', kind='intent', content='已解析用户请求', metadata={
        'task_spec': copy.deepcopy(state['task_spec']),
    })
    return state


def task_spec_from_partial(raw_spec: Dict[str, Any]) -> TaskSpec:
    nested_keys = (
        'system',
        'method',
        'job',
        'analysis',
        'runtime',
    )
    if any(isinstance(raw_spec.get(key), dict) for key in nested_keys):
        task_spec = task_spec_from_dict({
            key: raw_spec[key] for key in nested_keys if isinstance(raw_spec.get(key), dict)
        })
    else:
        task_spec = TaskSpec()

    for key in ('atom', 'basis', 'unit', 'charge', 'spin', 'symmetry'):
        if key in raw_spec:
            setattr(task_spec.system, key, raw_spec[key])

    if 'method' in raw_spec:
        task_spec.method.name = raw_spec['method']
    if 'xc' in raw_spec:
        task_spec.method.xc = raw_spec['xc']
    if 'restricted' in raw_spec:
        task_spec.method.restricted = raw_spec['restricted']
    if 'job' in raw_spec:
        task_spec.job.name = raw_spec['job']
    if 'outputs' in raw_spec:
        task_spec.analysis.outputs = list(raw_spec['outputs'])
    for key in ('max_cycle', 'conv_tol', 'verbose'):
        if key in raw_spec:
            setattr(task_spec.runtime, key, raw_spec[key])
    return task_spec


def spec_builder(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    raw_spec = state.get('task_spec') or {}
    if isinstance(raw_spec, TaskSpec):
        task_spec = raw_spec
    else:
        task_spec = task_spec_from_partial(raw_spec)
    state['task_spec'] = task_spec_to_dict(task_spec)
    append_log(state, 'info', 'workflow.spec_built', {
        'method': task_spec.method.name,
        'job': task_spec.job.name,
    })
    return state


def spec_validator(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state['task_spec'])
    errors = []
    questions = []

    if not task_spec.system.atom:
        errors.append('Missing molecular geometry in system.atom.')
        questions.append('请提供原子坐标，例如 O 0 0 0; H 0 0 0.96; H 0.92 0 -0.24')
    if not task_spec.system.basis:
        errors.append('Missing basis set in system.basis.')
        questions.append('请提供基组，例如 sto-3g、6-31g 或 cc-pvdz')
    if task_spec.method.name not in SUPPORTED_METHODS:
        errors.append('Only HF and DFT are supported in this MVP.')
    if task_spec.job.name not in SUPPORTED_JOBS:
        errors.append('Only single_point jobs are supported in this MVP.')
    if task_spec.method.name == 'dft' and not task_spec.method.xc:
        errors.append('DFT calculations require an xc functional.')
        questions.append('请提供泛函名称，例如 b3lyp 或 pbe')
    if task_spec.system.spin < 0:
        errors.append('Spin must be a non-negative integer.')

    if task_spec.method.restricted is None:
        task_spec.method.restricted = (task_spec.system.spin == 0)

    normalized_outputs = []
    for output in task_spec.analysis.outputs:
        if output not in normalized_outputs:
            normalized_outputs.append(output)
    task_spec.analysis.outputs = normalized_outputs or list(DEFAULT_ANALYSIS)

    state['task_spec'] = task_spec_to_dict(task_spec)
    state['validation_errors'] = errors
    state['clarification_questions'] = questions
    append_log(state, 'warning' if errors else 'info', 'workflow.spec_validated', {
        'error_count': len(errors),
        'question_count': len(questions),
    })
    if questions:
        append_message(state, role='assistant', kind='clarification', content='请补充缺失的计算信息', metadata={
            'questions': list(questions),
            'errors': list(errors),
        })
    return state


def _build_method_constructor(task_spec: TaskSpec) -> Tuple[str, List[str]]:
    '''Return the mean-field constructor and assignment lines for script export.'''
    settings = []
    method = task_spec.method
    if method.name == 'hf':
        constructor = 'scf.RHF(mol)' if method.restricted else 'scf.UHF(mol)'
    else:
        constructor = 'dft.RKS(mol)' if method.restricted else 'dft.UKS(mol)'
        settings.append("mf.xc = {0}".format(repr(method.xc)))
    settings.append('mf.max_cycle = {0}'.format(task_spec.runtime.max_cycle))
    if task_spec.runtime.conv_tol is not None:
        settings.append('mf.conv_tol = {0}'.format(task_spec.runtime.conv_tol))
    return constructor, settings


def generate_input_script(task_spec: TaskSpec) -> str:
    constructor, settings = _build_method_constructor(task_spec)
    helper_sources = (
        inspect.getsource(_extract_homo_lumo).rstrip(),
        inspect.getsource(_safe_to_list).rstrip(),
    )
    lines = [
        'from __future__ import annotations',
        'import io',
        'import json',
        'from contextlib import redirect_stdout',
        'from pyscf import dft, gto, scf',
        '',
        'mol = gto.M(',
        '    atom={0},'.format(repr(task_spec.system.atom)),
        '    basis={0},'.format(repr(task_spec.system.basis)),
        '    unit={0},'.format(repr(task_spec.system.unit)),
        '    charge={0},'.format(task_spec.system.charge),
        '    spin={0},'.format(task_spec.system.spin),
        '    symmetry={0},'.format(task_spec.system.symmetry),
        '    verbose={0},'.format(task_spec.runtime.verbose),
        ')',
        'mf = {0}'.format(constructor),
    ]
    lines.extend(settings)
    lines.extend(['', helper_sources[0], '', helper_sources[1], ''])
    lines.extend([
        'energy = mf.kernel()',
        'log_buffer = io.StringIO()',
        'with redirect_stdout(log_buffer):',
        '    mf.analyze()',
        'mo_occ = mf.mo_occ',
        'mo_energy = mf.mo_energy',
        'homo, lumo = _extract_homo_lumo(mo_energy, mo_occ)',
        'dm = mf.make_rdm1()',
        'dipole = _safe_to_list(mf.dip_moment(mol, dm))',
        'result = {',
        "    'converged': bool(mf.converged),",
        "    'energy': float(energy),",
        "    'homo': homo,",
        "    'lumo': lumo,",
        "    'gap': None if homo is None or lumo is None else lumo - homo,",
        "    'dipole': dipole,",
        "    'analysis_text': log_buffer.getvalue(),",
        '}',
        'print(json.dumps(result, indent=2))',
    ])
    return '\n'.join(lines) + '\n'


def input_generator(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state['task_spec'])
    state['generated_input'] = generate_input_script(task_spec)
    append_log(state, 'info', 'workflow.input_generated', {
        'line_count': len(state['generated_input'].splitlines()),
    })
    return state


def _extract_homo_lumo(mo_energy: Any, mo_occ: Any) -> Tuple[Optional[float], Optional[float]]:
    '''Extract HOMO and LUMO energies from restricted or unrestricted MO arrays.'''
    try:
        occ_ndim = mo_occ.ndim
    except AttributeError:
        occ_ndim = None

    if occ_ndim == 2:
        alpha_occ = mo_occ[0]
        alpha_energy = mo_energy[0]
        occupied = [float(energy) for energy, occ in zip(alpha_energy, alpha_occ) if occ > 0]
        virtual = [float(energy) for energy, occ in zip(alpha_energy, alpha_occ) if occ == 0]
    else:
        occupied = [float(energy) for energy, occ in zip(mo_energy, mo_occ) if occ > 0]
        virtual = [float(energy) for energy, occ in zip(mo_energy, mo_occ) if occ == 0]

    homo = occupied[-1] if occupied else None
    lumo = virtual[0] if virtual else None
    return homo, lumo


def _safe_to_list(value: Any) -> Any:
    '''Recursively convert arrays and nested containers to JSON-safe Python objects.'''
    if hasattr(value, 'tolist'):
        return _safe_to_list(value.tolist())
    if isinstance(value, dict):
        return {key: _safe_to_list(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_to_list(item) for item in value]
    return value


def _run_pyscf_task(task_spec: TaskSpec) -> Dict[str, Any]:
    '''Execute a PySCF MVP task and return JSON-serializable structured results.'''
    from pyscf import dft, gto, scf  # pylint: disable=import-outside-toplevel

    mol = gto.M(
        atom=task_spec.system.atom,
        basis=task_spec.system.basis,
        unit=task_spec.system.unit,
        charge=task_spec.system.charge,
        spin=task_spec.system.spin,
        symmetry=task_spec.system.symmetry,
        verbose=task_spec.runtime.verbose,
    )

    if task_spec.method.name == 'hf':
        mf = scf.RHF(mol) if task_spec.method.restricted else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if task_spec.method.restricted else dft.UKS(mol)
        mf.xc = task_spec.method.xc

    mf.max_cycle = task_spec.runtime.max_cycle
    if task_spec.runtime.conv_tol is not None:
        mf.conv_tol = task_spec.runtime.conv_tol

    raw_stdout = io.StringIO()
    with redirect_stdout(raw_stdout):
        energy = mf.kernel()
        mf.analyze()

    dm = mf.make_rdm1()
    dipole = mf.dip_moment(mol, dm)
    homo, lumo = _extract_homo_lumo(mf.mo_energy, mf.mo_occ)

    return {
        'converged': bool(mf.converged),
        'energy': float(energy),
        'homo': homo,
        'lumo': lumo,
        'gap': None if homo is None or lumo is None else lumo - homo,
        'dipole': dipole.tolist(),
        'mo_energy': _safe_to_list(mf.mo_energy),
        'mo_occ': _safe_to_list(mf.mo_occ),
        'analysis_text': raw_stdout.getvalue(),
    }


def runner(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if state['validation_errors']:
        state['execution_status'] = 'blocked'
        state['raw_stderr'] = '\n'.join(state['validation_errors'])
        append_log(state, 'warning', 'workflow.execution_blocked', {
            'errors': list(state['validation_errors']),
        })
        return state

    task_spec = task_spec_from_dict(state['task_spec'])
    append_log(state, 'info', 'workflow.execution_started', {
        'method': task_spec.method.name,
        'job': task_spec.job.name,
    })
    try:
        results = _run_pyscf_task(task_spec)
    except Exception as exc:  # pragma: no cover - exercised only with PySCF installed
        state['execution_status'] = 'failed'
        state['raw_stderr'] = str(exc)
        append_log(state, 'error', 'workflow.execution_failed', {
            'error': str(exc),
        })
        return state

    state['execution_status'] = 'succeeded'
    state['raw_stdout'] = results.get('analysis_text', '')
    state['structured_results'] = results
    append_log(state, 'info', 'workflow.execution_succeeded', {
        'converged': results.get('converged'),
    })
    return state


def result_extractor(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    results = state.get('structured_results') or {}
    extracted = {
        'energy': results.get('energy'),
        'homo': results.get('homo'),
        'lumo': results.get('lumo'),
        'gap': results.get('gap'),
        'dipole': results.get('dipole'),
        'converged': results.get('converged'),
    }
    state['structured_results'] = extracted
    append_log(state, 'info', 'workflow.results_extracted', {
        'keys': sorted(extracted.keys()),
    })
    return state


def result_analyst(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if state['execution_status'] != 'succeeded':
        if state['validation_errors']:
            state['analysis_summary'] = '任务未执行：' + '；'.join(state['validation_errors'])
        elif state['raw_stderr']:
            state['analysis_summary'] = '任务失败：{0}'.format(state['raw_stderr'])
        append_message(state, role='assistant', kind='summary', content=state['analysis_summary'])
        append_log(state, 'warning', 'workflow.summary_created', {
            'status': state['execution_status'],
        })
        return state

    results = state['structured_results'] or {}
    pieces = []
    if results.get('converged') is not None:
        pieces.append('SCF收敛={0}'.format('是' if results['converged'] else '否'))
    if results.get('energy') is not None:
        pieces.append('总能={0:.12f} Ha'.format(results['energy']))
    if results.get('homo') is not None and results.get('lumo') is not None:
        pieces.append(
            'HOMO={0:.6f} Ha, LUMO={1:.6f} Ha, 能隙={2:.6f} Ha'.format(
                results['homo'], results['lumo'], results['gap']
            )
        )
    if results.get('dipole') is not None:
        dipole = ', '.join('{0:.6f}'.format(value) for value in results['dipole'])
        pieces.append('偶极矩(Debye)=[{0}]'.format(dipole))
    state['analysis_summary'] = '；'.join(pieces)
    append_message(state, role='assistant', kind='summary', content=state['analysis_summary'], metadata={
        'structured_results': copy.deepcopy(results),
    })
    append_log(state, 'info', 'workflow.summary_created', {
        'status': state['execution_status'],
    })
    return state


def repair_or_retry(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    if state['execution_status'] != 'failed':
        return state
    if state['retry_count'] >= state.get('max_retries', 0):
        return state

    task_spec = task_spec_from_dict(state['task_spec'])
    lower_error = state.get('raw_stderr', '').lower()

    if 'converge' in lower_error or 'convergence' in lower_error:
        task_spec.runtime.max_cycle *= 2
        if task_spec.method.name == 'hf' and task_spec.system.spin != 0:
            task_spec.method.restricted = False
    state['retry_count'] += 1
    state['task_spec'] = task_spec_to_dict(task_spec)
    state['execution_status'] = 'pending'
    state['raw_stderr'] = ''
    append_log(state, 'info', 'workflow.retry_scheduled', {
        'retry_count': state['retry_count'],
        'max_cycle': task_spec.runtime.max_cycle,
    })
    append_message(state, role='system', kind='retry', content='任务将按保守参数重试', metadata={
        'retry_count': state['retry_count'],
        'runtime': copy.deepcopy(state['task_spec']['runtime']),
    })
    return state


def final_reporter(state: Dict[str, Any]) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state['task_spec'])
    state['final_report'] = {
        'channel': state.get('channel'),
        'task_spec': task_spec_to_dict(task_spec),
        'generated_input': state.get('generated_input'),
        'execution_status': state.get('execution_status'),
        'structured_results': state.get('structured_results'),
        'analysis_summary': state.get('analysis_summary'),
        'clarification_questions': state.get('clarification_questions'),
        'messages': copy.deepcopy(state.get('messages', [])),
    }
    append_log(state, 'info', 'workflow.finalized', {
        'status': state.get('execution_status'),
        'message_count': len(state.get('messages', [])),
        'log_count': len(state.get('logs', [])),
    })
    state['final_report']['logs'] = copy.deepcopy(state.get('logs', []))
    return state


def run_workflow(initial_state: Dict[str, Any]) -> Dict[str, Any]:
    state = intent_parser(initial_state)
    state = spec_builder(state)
    state = spec_validator(state)
    state = input_generator(state)
    state = runner(state)
    if state['execution_status'] == 'failed':
        repaired = repair_or_retry(state)
        if repaired['execution_status'] == 'pending':
            repaired = input_generator(repaired)
            repaired = runner(repaired)
        state = repaired
    state = result_extractor(state)
    state = result_analyst(state)
    state = final_reporter(state)
    return state


def execute_request(user_request: Any, *, channel: str = 'agent') -> Dict[str, Any]:
    return run_workflow(default_state(user_request, channel=channel))


def build_workflow():
    try:
        from langgraph.graph import END, StateGraph  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('langgraph is required to build the workflow graph') from exc

    graph = StateGraph(dict)
    graph.add_node('intent_parser', intent_parser)
    graph.add_node('spec_builder', spec_builder)
    graph.add_node('spec_validator', spec_validator)
    graph.add_node('input_generator', input_generator)
    graph.add_node('runner', runner)
    graph.add_node('repair_or_retry', repair_or_retry)
    graph.add_node('result_extractor', result_extractor)
    graph.add_node('result_analyst', result_analyst)
    graph.add_node('final_reporter', final_reporter)

    graph.set_entry_point('intent_parser')
    graph.add_edge('intent_parser', 'spec_builder')
    graph.add_edge('spec_builder', 'spec_validator')
    graph.add_edge('spec_validator', 'input_generator')
    graph.add_edge('input_generator', 'runner')
    graph.add_conditional_edges(
        'runner',
        lambda state: 'repair_or_retry' if state.get('execution_status') == 'failed' else 'result_extractor',
        {
            'repair_or_retry': 'repair_or_retry',
            'result_extractor': 'result_extractor',
        },
    )
    graph.add_conditional_edges(
        'repair_or_retry',
        lambda state: 'input_generator' if state.get('execution_status') == 'pending' else 'result_extractor',
        {
            'input_generator': 'input_generator',
            'result_extractor': 'result_extractor',
        },
    )
    graph.add_edge('result_extractor', 'result_analyst')
    graph.add_edge('result_analyst', 'final_reporter')
    graph.add_edge('final_reporter', END)
    return graph.compile()


def example_request() -> str:
    return json.dumps({
        'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        'basis': '6-31g',
        'method': 'dft',
        'xc': 'b3lyp',
        'job': 'single_point',
        'outputs': ['energy', 'homo_lumo', 'dipole', 'mulliken'],
    })


def main() -> None:
    final_state = execute_request(example_request())
    print(json.dumps(final_state['final_report'], indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
