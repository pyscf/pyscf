#!/usr/bin/env python

import importlib
import json
import pathlib
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BACKEND = importlib.import_module('Pyscf_agent.backend')
CLI = importlib.import_module('Pyscf_agent.cli')
MODULE = importlib.import_module('Pyscf_agent.main')
WEB = importlib.import_module('Pyscf_agent.web')


class KnownValues(unittest.TestCase):
    def test_spec_builder_and_validator(self):
        state = MODULE.default_state('atom: H 0 0 0; H 0 0 0.74\nbasis: sto-3g\nmethod: hf')
        state = MODULE.intent_parser(state)
        state = MODULE.spec_builder(state)
        state = MODULE.spec_validator(state)

        self.assertEqual(state['validation_errors'], [])
        self.assertEqual(state['task_spec']['system']['basis'], 'sto-3g')
        self.assertTrue(state['task_spec']['method']['restricted'])

    def test_validator_requests_missing_fields(self):
        state = MODULE.default_state('Run a DFT calculation')
        state = MODULE.intent_parser(state)
        state = MODULE.spec_builder(state)
        state = MODULE.spec_validator(state)

        self.assertTrue(state['validation_errors'])
        self.assertTrue(state['clarification_questions'])

    def test_generate_input_script(self):
        task_spec = MODULE.TaskSpec(
            system=MODULE.SystemSpec(
                atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                basis='6-31g',
            ),
            method=MODULE.MethodSpec(name='dft', restricted=True, xc='b3lyp'),
        )

        script = MODULE.generate_input_script(task_spec)
        self.assertIn("mf = dft.RKS(mol)", script)
        self.assertIn("mf.xc = 'b3lyp'", script)
        self.assertIn("basis='6-31g'", script)

    def test_parse_user_request_merges_json_and_prompt_hints(self):
        parsed = BACKEND.parse_user_request(json.dumps({
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'request': 'Run a DFT calculation and report dipole',
        }))

        self.assertEqual(parsed['method'], 'dft')
        self.assertEqual(parsed['job'], 'single_point')
        self.assertIn('dipole', parsed['outputs'])

    def test_result_analyst_summary(self):
        state = MODULE.default_state('irrelevant')
        state['execution_status'] = 'succeeded'
        state['structured_results'] = {
            'converged': True,
            'energy': -75.983948,
            'homo': -0.281,
            'lumo': 0.112,
            'gap': 0.393,
            'dipole': [0.0, 0.0, 1.85],
        }

        state = MODULE.result_analyst(state)
        self.assertIn('总能', state['analysis_summary'])
        self.assertIn('能隙', state['analysis_summary'])
        self.assertIn('偶极矩', state['analysis_summary'])

    def test_retry_policy_doubles_max_cycle(self):
        task_spec = MODULE.TaskSpec(
            system=MODULE.SystemSpec(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g'),
            method=MODULE.MethodSpec(name='hf', restricted=True),
        )
        state = MODULE.default_state('irrelevant')
        state['task_spec'] = MODULE.task_spec_to_dict(task_spec)
        state['execution_status'] = 'failed'
        state['raw_stderr'] = 'SCF failed to converge'

        state = MODULE.repair_or_retry(state)
        self.assertEqual(state['execution_status'], 'pending')
        self.assertEqual(state['task_spec']['runtime']['max_cycle'], 100)
        self.assertEqual(state['retry_count'], 1)

    def test_backend_records_structured_messages_and_logs(self):
        final_state = BACKEND.execute_request('Run a DFT calculation', channel='cli')
        report = final_state['final_report']

        self.assertEqual(report['channel'], 'cli')
        self.assertTrue(report['messages'])
        self.assertTrue(report['logs'])
        self.assertEqual(report['messages'][0]['role'], 'user')
        self.assertEqual(report['messages'][0]['channel'], 'cli')
        self.assertEqual(report['messages'][1]['kind'], 'intent')
        self.assertEqual(report['messages'][-1]['kind'], 'summary')
        self.assertIn('workflow.initialized', [item['event'] for item in report['logs']])
        self.assertIn('workflow.finalized', [item['event'] for item in report['logs']])

    def test_cli_returns_json_report(self):
        output = StringIO()
        with redirect_stdout(output):
            return_code = CLI.main([
                'Run a DFT calculation',
                '--pretty',
            ])

        payload = output.getvalue().strip()
        report = json.loads(payload)
        self.assertEqual(return_code, 0)
        self.assertEqual(report['channel'], 'cli')
        self.assertIn('messages', report)
        self.assertIn('logs', report)

    def test_web_api_uses_shared_backend(self):
        status, headers, body = WEB.handle_api_request(json.dumps({
            'request': 'Run a DFT calculation',
        }).encode('utf-8'))

        report = json.loads(body.decode('utf-8'))
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(report['channel'], 'web')
        self.assertIn('messages', report)
        self.assertIn('logs', report)

    def test_web_api_accepts_structured_form_payload(self):
        status, headers, body = WEB.handle_api_request(json.dumps({
            'request': 'Please focus on dipole analysis',
            'task_spec': {
                'atom': 'O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                'basis': '6-31g',
                'method': 'dft',
                'xc': 'b3lyp',
                'job': 'single_point',
                'outputs': ['energy', 'dipole'],
            },
        }).encode('utf-8'))

        report = json.loads(body.decode('utf-8'))
        self.assertEqual(status, 200)
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(report['task_spec']['method']['name'], 'dft')
        self.assertEqual(report['task_spec']['method']['xc'], 'b3lyp')
        self.assertEqual(report['task_spec']['analysis']['outputs'], ['energy', 'dipole'])

    def test_web_index_contains_frontend(self):
        page = WEB.build_index_html()
        self.assertIn('PySCF Agent 网页界面', page)
        self.assertIn('/api/run', page)
        self.assertIn('select id="basis"', page)
        self.assertIn('select id="method"', page)
        self.assertIn('select id="xc"', page)
        self.assertIn('div id="conversation"', page)
        self.assertIn('pre id="input-preview"', page)
        self.assertIn('textarea id="request"', page)
        self.assertIn('button id="run"', page)
        self.assertNotIn('pre id="logs"', page)


if __name__ == '__main__':
    unittest.main()
