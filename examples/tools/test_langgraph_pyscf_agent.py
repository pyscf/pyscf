#!/usr/bin/env python

import importlib.util
import pathlib
import sys
import unittest


MODULE_PATH = pathlib.Path(__file__).with_name('09-langgraph_pyscf_agent.py')
SPEC = importlib.util.spec_from_file_location('langgraph_pyscf_agent_example', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


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
        self.assertIn('Gap', state['analysis_summary'])
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


if __name__ == '__main__':
    unittest.main()
