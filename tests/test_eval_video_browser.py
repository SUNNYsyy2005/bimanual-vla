import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from server_4090.eval_video_browser import VideoBrowser,publish_index,group_id


class Args(dict):
    def getlist(self,key):return self.get(key,[])


class VideoBrowserTest(unittest.TestCase):
    def test_groups_need_no_video_index_and_empty_selection_never_reads_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'browse-groups.json').write_text(json.dumps(dict(groups=[],total=0,updated_at=1)))
            browser=VideoBrowser(root)
            self.assertTrue(browser.groups()['ready'])
            self.assertTrue(browser.page(Args())['selection_required'])
            self.assertFalse((root/'browse-index.json').exists())

    def rows(self,root):
        return [dict(root=str(root),relative_path=f'{task}/{exp}/{i}.mp4',name=f'{i}.mp4',video_source='libero-safety',
                     task=task,experiment=exp,run_id='run',mtime=1000-i,success='success' if i%2 else 'failed',
                     safe_success='failed',phase='eval',episode_record={'large':'unused'},has_events=i==0)
                for task,exp,n in [('tsa-basket','a',50),('tsa-basket','b',3),('tsa-microwave','a',4)] for i in range(n)]

    def test_exact_multi_selection_balanced_pages_and_filter_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=self.rows(root);publish_index(root,rows,{})
            browser=VideoBrowser(root);scopes=[group_id(rows[0]),group_id(rows[50])]
            first=browser.page(Args(scope=scopes,limit='4'))
            self.assertEqual(first['total'],53)
            self.assertEqual([v['experiment'] for v in first['videos']],['a','b','a','b'])
            self.assertEqual({v['task'] for v in first['videos']},{'tsa-basket'})
            second=browser.page(Args(scope=scopes,limit='4',offset='4'))
            self.assertFalse({v['id'] for v in first['videos']}&{v['id'] for v in second['videos']})
            filtered=browser.page(Args(scope=scopes,success='success'))
            self.assertEqual(filtered['total'],26)
            self.assertNotIn('episode_record',first['videos'][0])
            with self.assertRaises(ValueError):browser.page(Args(scope=['bad']))

    def test_synced_remote_dedup_delete_and_index_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=self.rows(root)[0]
            row['provenance']=dict(storage_source='remote',root='/remote',relative_path='a.mp4')
            remote=dict(row,root='/remote',relative_path='a.mp4')
            publish_index(root,[row],{'remote':{'videos':[remote]}})
            browser=VideoBrowser(root);args=Args(scope=[group_id(row)])
            page=browser.page(args);self.assertEqual(page['total'],1)
            browser.discard([page['videos'][0]['id']]);self.assertEqual(browser.page(args)['total'],0)
            new=dict(row,relative_path='new.mp4')
            publish_index(root,[new],{});self.assertEqual(browser.page(args)['total'],1)


if __name__=='__main__':unittest.main()
