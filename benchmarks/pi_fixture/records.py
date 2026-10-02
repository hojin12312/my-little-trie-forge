"""Legacy cache record parser; public return contract must stay stable."""
import json
def parse_record(text):
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('record must be an object')
    if 'key' not in data or not isinstance(data['key'], str):
        raise ValueError('key must be a string')
    if 'value' not in data:
        raise ValueError('value is required')
    return (data['key'], data['value'])
class JsonStore:
    def __init__(self, cache):
        self.cache = cache
    def ingest(self, text):
        key, value = parse_record(text)
        self.cache.put(key, value)
        return key
