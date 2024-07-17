import os
import glob

def count_python_lines(directory):
    total_lines = 0
    # 디렉토리 내의 모든 .py 파일 찾기
    python_files = glob.glob(os.path.join(directory, '**', '*.py'), recursive=True)
    
    for file_path in python_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            total_lines += len(lines)
    
    return total_lines


if __name__ == "__main__" :
    # 프로젝트 디렉토리 경로를 입력하세요
    project_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Project directory: ", project_directory)
    total_lines = count_python_lines(project_directory)
    print(f"Total number of lines in the Python project: {total_lines}")
