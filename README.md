- This repository contains an end-to-end system simulation for SISO OFDM system analysis
- For the data retrieval the following equalizers have been implemented:
     - zero-forcing
     - matched-filtering
     - MMSE
       
- For large SNR the MMSE reduces to the zero-forcing and in the case of low SNR it approaches the matched-filtering equalizer 
- In the following example the impulse response of the channel has been set as an FIR system with 2 taps: h = [1,0.1j]
- The SNR is 30 dB and the constellation is based on 16-QAM
- The number of sub-carriers is 128 out of which 9 of them are pilot signals that have been used to estimate the channel



![Channel_IPR](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/fdeada23-e152-40d9-9c1f-b8873018c63f)


![Constellation](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/7cdf77a5-7ef3-42cd-bf1c-c2d916cd7156)




 - Following is another example based on a differrnt wireless channel with 4 taps: h = [1,   j,   0.1 + 0.5j,   0.2 + 0.2j]

![Channel_IPR](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/4529b89e-d6f7-4aa0-9005-c0d6ee9e1c23)







![Constellation](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/00448f67-4562-4cad-8bd7-d06afa1f709b)
