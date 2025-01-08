import numpy as np
import matplotlib.pyplot as plt

# 1. 랜덤 데이터 생성
def generate_data(bits=1000):
    return np.random.randint(0, 2, bits)  # 0과 1로 이루어진 랜덤 데이터 생성

# 2. AM 변조 함수
def am_modulation(data, carrier_freq, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sampled_data = np.repeat(data, len(t) // len(data))  # 데이터 크기를 반송파 크기에 맞춤
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    modulated_signal = (1 + sampled_data[:len(t)]) * carrier
    return t, modulated_signal

# 3. FM 변조 함수
def fm_modulation(data, carrier_freq, sampling_rate, duration, freq_dev=50):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sampled_data = np.repeat(data, len(t) // len(data))  # 데이터 크기를 반송파 크기에 맞춤
    integral = np.cumsum(sampled_data[:len(t)]) / sampling_rate
    modulated_signal = np.cos(2 * np.pi * carrier_freq * t + 2 * np.pi * freq_dev * integral)
    return t, modulated_signal

# 4. QAM 변조 함수
def qam_modulation(data, carrier_freq, sampling_rate, duration):
    t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
    sampled_data = np.repeat(data, len(t) // len(data))[:len(t)]  # 데이터 크기를 반송파 크기에 맞춤
    data_i = sampled_data[0::2]  # 실수부
    data_q = sampled_data[1::2]  # 허수부
    carrier_i = np.cos(2 * np.pi * carrier_freq * t[:len(data_i)])
    carrier_q = np.sin(2 * np.pi * carrier_freq * t[:len(data_q)])
    modulated_signal = data_i * carrier_i - data_q * carrier_q
    return t[:len(data_i)], modulated_signal

# 5. AWGN 추가 함수
def add_awgn(signal, SNR_dB):
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (SNR_dB / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal

# 6. 복조 및 BER 계산 함수
def calculate_ber(original_data, received_data):
    errors = np.sum(original_data != received_data)
    ber = errors / len(original_data)
    return ber

# 7. 실험 시뮬레이션
def simulate():
    # 기본 설정
    bits = 1000
    sampling_rate = 10000
    carrier_freq = 1000
    duration = 1
    SNR_values = [0, 5, 10, 15, 20]  # SNR (dB)

    # 데이터 생성
    data = generate_data(bits)

    # 결과 저장용 리스트
    results = {"AM": [], "FM": [], "QAM": []}

    for SNR in SNR_values:
        # AM 실험
        t, am_signal = am_modulation(data, carrier_freq, sampling_rate, duration)
        noisy_am = add_awgn(am_signal, SNR)
        # 간단화된 복조를 적용 (이 부분은 이후 확장 가능)
        demod_am = np.round(noisy_am).astype(int)[:len(data)]  
        ber_am = calculate_ber(data[:len(demod_am)], demod_am)
        results["AM"].append(ber_am)

        # FM 실험
        t, fm_signal = fm_modulation(data, carrier_freq, sampling_rate, duration)
        noisy_fm = add_awgn(fm_signal, SNR)
        demod_fm = np.round(noisy_fm).astype(int)[:len(data)]  
        ber_fm = calculate_ber(data[:len(demod_fm)], demod_fm)
        results["FM"].append(ber_fm)

        # QAM 실험
        t, qam_signal = qam_modulation(data, carrier_freq, sampling_rate, duration)
        noisy_qam = add_awgn(qam_signal, SNR)
        demod_qam = np.round(noisy_qam).astype(int)[:len(data)]  
        ber_qam = calculate_ber(data[:len(demod_qam)], demod_qam)
        results["QAM"].append(ber_qam)

    return SNR_values, results

# 8. 결과 시각화
def plot_results(SNR_values, results):
    plt.figure(figsize=(10, 6))
    plt.plot(SNR_values, results["AM"], label="AM", marker="o")
    plt.plot(SNR_values, results["FM"], label="FM", marker="o")
    plt.plot(SNR_values, results["QAM"], label="QAM", marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("SNR vs BER for Different Modulation Techniques")
    plt.legend()
    plt.grid(True)
    plt.show()

# 9. 실행
SNR_values, results = simulate()
plot_results(SNR_values, results)
