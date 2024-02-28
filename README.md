- This repository contains an end-to-end system simulation for SISO OFDM system analysis
- For the data retrieval the following equalizers have been implemented:
     - zero-forcing
     - matched-filtering
     - MMSE
       
- For large SNR the MMSE reduces to the zero-forcing and in the case of low SNR it approaches the matched-filtering equalizer 
- In the following example the impulse response of the channel has been set as an FIR system with 2 taps: h = [1,0.1j]
- The SNR is 30 dB and the constellation is based on 16-QAM
- The number of sub-carriers is 128 out of which 9 of them are pilot signals that have been used to estimate the channel


![Channel_IPR](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/1a6de366-d331-4496-b09f-37c30c6f78bc)




![Constellation](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/d6fa784e-8781-4a6f-800c-bc312043e1cc)



 - Following is another example based on a differrnt wireless channel with 4 taps: h = [1,   j,   0.1 + 0.5j,   0.2 + 0.2j]

![Channel_IPR](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/e78c72fa-da38-4f48-b9e2-61fdb5c6e5d3)


![Constellation](https://github.com/Shahrokh-Hamidi/OFDM-End-to-End-simulation/assets/156338354/3435e4a3-1acc-4a60-a628-8374ede2ff20)
