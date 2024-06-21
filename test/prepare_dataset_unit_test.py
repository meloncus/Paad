import unittest
from unittest.mock import patch

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from python.prepare_dataset import get_data_paths_and_labels_from_edge_dir

class TestPrepareDataset(unittest.TestCase):
    @patch('os.listdir')
    def test_get_data_paths_and_labels_from_edge_dir(self, mock_listdir):
        # 가상의 파일 리스트를 정의
        mock_listdir.return_value = ['normal_01.wav', 'abnormal_01.wav', 'other_file.txt']
        
        # 함수 실행
        data_path = '/path/to/data/'
        data_dict, label_dict = get_data_paths_and_labels_from_edge_dir(data_path)
        
        # 예상 결과
        expected_data_dict = {
            'normal': [data_path + 'normal_01.wav'],
            'abnormal': [data_path + 'abnormal_01.wav'],
            'other': []
        }
        expected_label_dict = {
            data_path + 'normal_01.wav': 1,
            data_path + 'abnormal_01.wav': -1,
            data_path + 'other_file.txt': 0
        }
        
        # 결과 검증
        self.assertEqual(data_dict, expected_data_dict)
        self.assertEqual(label_dict, expected_label_dict)

if __name__ == '__main__':
    unittest.main()