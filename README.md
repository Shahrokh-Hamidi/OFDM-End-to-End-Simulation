- This repository contains an end-to-end system simulation for SISO OFDM system analysis
- For the channel estimation the fowlloing have been implemented:
     - zero forcing
     - matched filtering
     - MMSE

- In the following example the impulse response of the channel has been set as an FIR system with 4 taps: h = [1,   j,   0.1 + 0.5j,   0.2 + 0.2j]
- The SNR is 30 dB and the constellation is based on 16-QAM
- The number of sub-carriers is 64 out of which 9 of them are pilot signals that have been used to estimate the channel


![channel_IPR](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/b6c52572-c7b2-4b86-9660-9d1f35a93841)





![constelation](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/1f23f1ef-795d-4de0-a0e4-d8985b7f7944)

