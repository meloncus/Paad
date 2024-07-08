# TODO : 이 코드는 오디오 파일을 로드하는 함수를 정의한 것입니다. 이 함수는 torchaudio.load 함수를 사용하여 오디오 파일을 로드하고, PyTorch 텐서로 변환하여 반환합니다. 이 코드는 다음과 같은 몇 가지 제한 사항이 있습니다.
# 파일 형식 지원: torchaudio.load 함수는 다양한 오디오 파일 형식을 지원하지만, 모든 형식을 지원하는 것은 아닙니다. 특정 형식의 파일을 로드하려고 할 때 호환성 문제가 발생할 수 있습니다.
# 샘플 레이트 변환: 입력 파일의 샘플 레이트가 모델이나 처리 파이프라인에서 요구하는 샘플 레이트와 다를 수 있습니다. 이 코드는 샘플 레이트 변환을 처리하지 않으므로, 필요한 경우 추가적인 샘플 레이트 변환 로직이 필요합니다.
# 멀티채널 오디오 처리: 이 코드는 멀티채널 오디오를 모노로 변환합니다. 그러나 특정 채널만 선택하거나, 모든 채널을 개별적으로 처리하고 싶은 경우에는 이를 지원하지 않습니다. channel 매개변수는 현재 사용되지 않으며, 특정 채널을 선택하는 기능을 구현할 필요가 있습니다.
# 메모리 사용량: 대용량 오디오 파일을 로드할 때, 이 코드는 전체 파일을 메모리에 로드합니다. 대규모 데이터셋을 처리할 때 메모리 사용량이 문제가 될 수 있으며, 이를 위해 스트리밍이나 배치 처리 방식을 고려할 수 있습니다.
# 오류 처리: 이 코드는 파일 로드 과정에서 발생할 수 있는 예외나 오류(예: 파일이 존재하지 않거나, 손상된 파일)를 명시적으로 처리하지 않습니다. 오류 처리 로직을 추가하여 더 견고한 코드를 작성할 수 있습니다.
# 정규화 옵션: normalize=True 옵션은 오디오 데이터를 [-1.0, 1.0] 범위로 정규화합니다. 이는 대부분의 경우에 유용하지만, 원본 데이터의 동적 범위를 유지하고 싶은 경우에는 이 옵션을 비활성화할 수 있습니다. 사용자가 이 옵션을 더 세밀하게 제어할 수 있도록 하는 것이 좋습니다.
# 성능 최적화: librosa.to_mono를 사용하여 NumPy 배열로 변환한 후 다시 PyTorch 텐서로 변환하는 과정은 추가적인 계산 비용을 발생시킵니다. 특히 대규모 데이터셋을 처리할 때 이러한 변환으로 인한 오버헤드가 문제가 될 수 있습니다. 가능하다면, 이러한 변환 과정을 최소화하거나, 멀티쓰레딩/멀티프로세싱을 활용한 병렬 처리를 고려할 수 있습니다.

import torchaudio
import torch

import librosa


def load_audio(filename, normalize=True, channel=0) :
    '''
    load audio file and return waveform and sample rate

    input
    filename : str
        target .wav file
    normalize : bool (default : True)
    channel : int (default : 0)

    output
    waveform : torch.tensor
    sample_rate : int
    '''

    waveform, sample_rate = torchaudio.load(filename, normalize=normalize)

    # if waveform is stereo, number of channels is 2
    num_channels, num_frames = waveform.shape
    if num_channels != 1:
        x = librosa.to_mono(waveform.numpy())
        waveform = torch.tensor(x).float() # convert to torch.tensor and float
        waveform = waveform.unsqueeze(0) # add channel dimension [1, num_frames]
    return waveform, sample_rate