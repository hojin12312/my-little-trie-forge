import unittest
from tempfile import TemporaryDirectory
from queue_cache import Queue, DiskCache
from runner import Runner
from records import parse_record, JsonStore
class Tests(unittest.TestCase):
    def test_cache_roundtrip(self):
        with TemporaryDirectory() as root:
            cache = DiskCache(root)
            cache.put('report', {'ok': True})
            self.assertEqual(cache.get('report'), {'ok': True})
    def test_completed_job(self):
        with TemporaryDirectory() as root:
            cache = DiskCache(root)
            queue = Queue(cache)
            job = queue.submit('report')
            queue.complete(job, 'new')
            self.assertEqual(cache.get('report'), 'new')
            self.assertNotIn('report', queue.active)
    def test_runner_order(self):
        self.assertEqual(Runner(2).run([lambda: 3, lambda: 7]), [3, 7])
    def test_record_contract(self):
        self.assertEqual(parse_record('{"key":"report","value":4}'), ('report', 4))
        for text in ['[]', '{}', '{"key":5,"value":4}', '{"key":"report"}']:
            with self.assertRaises(ValueError):
                parse_record(text)
if __name__ == '__main__':
    unittest.main()
