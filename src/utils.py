import os
import yaml


def get_files_under_directory(home_dir):
    return [os.path.join(root, file) for root, _, files in os.walk(home_dir) for file in files]


def get_files_from_segments(data_dir):
    segments_path = os.path.join(data_dir, 'txt/segments')
    file_list = []
    with open(segments_path, 'r') as f:
        for line in f.readlines():
            file_name = line.strip().split(' ')[0]
            file_path = os.path.join(data_dir, 'wav', file_name)
            file_list.append(file_path)
    return file_list


def get_files_from_yaml(data_dir):
    yaml_path = os.path.join(data_dir, 'txt', os.path.basename(os.path.normpath(data_dir)) + '.yaml')
    with open(yaml_path, 'r') as stream:
        try:
            yaml_instances = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise Exception(exc)
    file_list = [os.path.join(data_dir, 'wav', inst['wav'] + '.wav') for inst in yaml_instances]
    return file_list


def get_transcription_from_file(data_dir, lang):
    transcription_path = os.path.join(data_dir, 'txt/', os.path.basename(os.path.normpath(data_dir)) + '.' + lang)
    transcriptions = []
    with open(transcription_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            transcriptions.append(line)
    return transcriptions


if __name__ == '__main__':
    data_dir = 'data/swa-eng/valid/'
    get_files_from_yaml(data_dir)