import copy
import unittest
import validate_measurement as v
class Contract(unittest.TestCase):
    def good_rows(self):return [{'cell_id':k} for k in sorted(v.EXPECTED)]
    def good_window(self):
        rows=[{'event':'ready','lane':i} for i in range(4)]+[{'event':'ALL_READY_RELEASE','at':10,'B':4,'R':4}]
        for i in range(20):
            rows.append({'event':'decode_cycle','cycle':i,'begin':10+i*.1,'end':10+(i+1)*.1,'wall_seconds':.1,
                         'B':4,'boundary_completion':False,'primary':True,'confirmed':8,'drafted':28,'accepted':4,
                         'lane_tokens':[{'tokens':[1,2],'drafted':7,'accepted':1} for _ in range(4)]})
        return rows
    def test_coverage(self):v.coverage(self.good_rows())
    def test_missing(self):
        with self.assertRaisesRegex(ValueError,'missing'):v.coverage(self.good_rows()[:-1])
    def test_duplicate(self):
        with self.assertRaisesRegex(ValueError,'duplicate'):v.coverage(self.good_rows()+self.good_rows()[:1])
    def test_input_budget(self):v.budget(261568,512,64,261568)
    def test_input_mismatch(self):
        with self.assertRaisesRegex(ValueError,'mismatch'):v.budget(2048,512,64,2047)
    def test_input_overflow(self):
        with self.assertRaisesRegex(ValueError,'budget'):v.budget(262144,512,64,262144)
    def test_counter_reset(self):
        with self.assertRaisesRegex(ValueError,'negative'):v.counter_delta({'tokens':10},{'tokens':9},'a','a')
    def test_mixed_instance(self):
        with self.assertRaisesRegex(ValueError,'instance'):v.counter_delta({'tokens':10},{'tokens':12},'a','b')
    def test_zero_draft(self):self.assertIsNone(v.acceptance(0,0))
    def test_bonus_not_accepted(self):self.assertEqual(v.acceptance(4,7,bonus=1),4/7)
    def test_accepted_over_draft(self):
        with self.assertRaises(ValueError):v.acceptance(8,7)
    def test_padding(self):
        with self.assertRaisesRegex(ValueError,'padded'):v.acceptance(4,7,inactive=1)
    def test_http_boundary(self):
        events=[{'at':1,'event':{'choices':[{'delta':{'role':'assistant'}}]}},
                {'at':2,'event':{'choices':[{'delta':{'content':''}}]}},
                {'at':4,'event':{'choices':[{'delta':{'content':'multiple tokens in one chunk'}}]}}]
        self.assertEqual(v.first_content(events,0),4)
    def test_native_vs_wave(self):
        r=v.phase_rates(2048,512,2,4,10);self.assertEqual(r,{'PP_native':1024,'D_native':128,'full_wave':51.2})
    def test_valid_width(self):self.assertEqual(len(v.validate_d_window(self.good_window(),4)),20)
    def test_C_not_B(self):
        e=self.good_window();e[4]['B']=2
        with self.assertRaisesRegex(ValueError,'B/R'):v.validate_d_window(e,4)
    def test_shape_transition(self):
        e=self.good_window();e[8]['B']=3
        with self.assertRaisesRegex(ValueError,'shape'):v.validate_d_window(e,4)
    def test_prefill_overlap(self):
        e=self.good_window()+[{'event':'prefill','end':11}]
        with self.assertRaisesRegex(ValueError,'prefill'):v.validate_d_window(e,4)
    def test_time_mismatch(self):
        e=self.good_window();e[8]['wall_seconds']=.2
        with self.assertRaisesRegex(ValueError,'interval'):v.validate_d_window(e,4)
    def test_cherry_pick(self):
        e=self.good_window();e[8]['primary']=False
        with self.assertRaisesRegex(ValueError,'contiguous'):v.validate_d_window(e,4)
    def test_dummy_final(self):
        with self.assertRaisesRegex(ValueError,'dummy'):v.final_input({'data_kind':'layout_dummy'})
    def valid_final(self):return {'data_kind':'measured_result','cells':[dict(r,sampling={'temperature':0,'thinking':False,'output_cap':512},cached_tokens=0,peak_scope='fresh_process_lifetime') for r in self.good_rows()]}
    def test_cold_rejection(self):
        d=self.valid_final();d['cells'][0]['cached_tokens']=32
        with self.assertRaisesRegex(ValueError,'cold'):v.final_input(d)
    def test_historical_rejection(self):
        d=self.valid_final();d['cells'][0]['sampling']['temperature']=1
        with self.assertRaisesRegex(ValueError,'historical'):v.final_input(d)
    def test_peak_scope(self):
        d=self.valid_final();d['cells'][0]['peak_scope']='total Metal+RSS'
        with self.assertRaisesRegex(ValueError,'peak'):v.final_input(d)
    def test_failed_preserved(self):self.assertEqual(v.summarize_statuses([{'status':s} for s in ['PASS','ERROR','CANCELLED']]),{'PASS':1,'ERROR':1,'CANCELLED':1})
    def test_bilingual_units(self):
        with self.assertRaisesRegex(ValueError,'bilingual'):v.bilingual_equal({'peak':48,'unit':'GiB'},{'peak':48,'unit':'GB'})
if __name__=='__main__':unittest.main()
