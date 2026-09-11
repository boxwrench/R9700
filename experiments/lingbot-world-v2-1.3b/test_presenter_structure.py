"""Structural guard for the Presenter/display wiring (no GPU, no torch).

Regression test for an edit collision that once truncated
Presenter.__init__ mid-body: py_compile stayed green while the live path
crashed with AttributeError on first present. Parses the source with ast
instead of importing it, so this runs anywhere.
"""
import ast
import os
import unittest

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   'interactive_world.py')

# Attributes the live present path requires on every Presenter instance.
REQUIRED_INIT_ATTRS = {
    'archive', 'archive_thread', 'q', 'results', 'frames_dir',
    'upscale2', 'upscale2_sync', 'up2_seq', 'up2_frame_counter',
    'display_q', 'display_thread', 'present_server', 'up2_pub', 'up2_lock',
}


def _methods():
    tree = ast.parse(open(SRC).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'Presenter':
            return {f.name: f for f in node.body
                    if isinstance(f, ast.FunctionDef)}
    raise AssertionError('Presenter class not found')


def _stores(func):
    return {n.attr for n in ast.walk(func)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == 'self' and isinstance(n.ctx, ast.Store)}


class PresenterStructureTest(unittest.TestCase):
    def test_init_assigns_live_path_attributes(self):
        init_attrs = _stores(_methods()['__init__'])
        missing = REQUIRED_INIT_ATTRS - init_attrs
        self.assertFalse(missing, f'Presenter.__init__ never assigns: {missing}')

    def test_display_worker_sets_no_instance_state(self):
        # The display thread must be a pure function of its queue items;
        # per-frame re-created queues/threads would be a lifecycle bug.
        worker_attrs = _stores(_methods()['_display_run'])
        self.assertFalse(worker_attrs,
                         f'_display_run must not assign self.*: {worker_attrs}')

    def test_present_server_lifecycle_methods_exist(self):
        methods = _methods()
        for name in ('_display_put', '_display_run'):
            self.assertIn(name, methods)


if __name__ == '__main__':
    unittest.main()
