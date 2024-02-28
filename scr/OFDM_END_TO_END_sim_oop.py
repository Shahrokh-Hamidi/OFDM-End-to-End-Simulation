# Created by Shahrokh Hamidi
# PhD., Electrical & Computer Engineering




import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib


class Mapping:


    def __init__(self):
        pass


    @staticmethod
    def map_bit2symb():
    
        dic_ =  {
            (0,0,0,0) : -3-3j,
            (0,0,0,1) : -3-1j,
            (0,0,1,0) : -3+3j,
            (0,0,1,1) : -3+1j,
            (0,1,0,0) : -1-3j,
            (0,1,0,1) : -1-1j,
            (0,1,1,0) : -1+3j,
            (0,1,1,1) : -1+1j,
            (1,0,0,0) :  3-3j,
            (1,0,0,1) :  3-1j,
            (1,0,1,0) :  3+3j,
            (1,0,1,1) :  3+1j,
            (1,1,0,0) :  1-3j,
            (1,1,0,1) :  1-1j,
            (1,1,1,0) :  1+3j,
            (1,1,1,1) :  1+1j
        }
        
        return dic_
    
    @staticmethod
    def map_symb2bit():

        dic_ = {symb: bit for bit, symb in Mapping.map_bit2symb().items()}
        return dic_
    

class dataGen:

    def __init__(self):

        pass


    @staticmethod
    def DATA_generation(len):
        return np.random.binomial(n = 1, p = 0.5, size = (len, ))





class Utils(Mapping):

    def __init__(self):

        pass

    
    def Serial_to_Parallel(self, data):
        return data.reshape(-1, num_bits_per_symb)


    def map_bit_to_symb(self, data):

        return np.array([Mapping.map_bit2symb()[tuple(v)]  for v in data])


    def map_symb_to_bit(self, data):

        return np.array([Mapping.map_symb2bit()[v]  for v in data])



    def apply_ifft(self, data):

        return np.fft.ifft(data)


    def adding_CP(self, data, CP):
        
        return np.hstack((data[-CP:], data))
    
  
    def remove_CP(self, data):

        return data[num_cyclic_prefix: num_cyclic_prefix + num_subcarriers]


    def apply_fft(self, data):

        return np.fft.fft(data)






class Channel:

    def __init__(self):
        
        pass

    
    def channel_(self, data, h):

        return np.convolve(data, h, mode = 'same')

    
    def channel_estimation(self, data):

        pilot_value_est = data[pilot_index]
        return pilot_value_est/pilot_value


    def channel_estimation_interpolation(self, data):

        h_amp = scipy.interpolate.interp1d(pilot_index, abs(data), kind='cubic')(subcarrier_index)
        h_phase = scipy.interpolate.interp1d(pilot_index, np.angle(data), kind='cubic')(subcarrier_index)
        
        return h_amp*np.exp(1j*h_phase)



    def channel_impulse_response_visualization(self, data):

        H = np.fft.fft(channel_impulse_response, num_subcarriers)
        plt.plot(abs(data), 'k', label = 'estimated impulse response', linewidth = 2)
        plt.plot(abs(H), 'r', label = 'true impulse response', linewidth = 2)
        plt.xlabel('subcarrier index', fontsize = 16)
        plt.ylabel('|H(f)|', fontsize = 16)
        plt.title('channel impulse response', fontsize = 14)
        plt.legend()
        matplotlib.rc('xtick', labelsize=16) 
        matplotlib.rc('ytick', labelsize=16) 

        plt.grid()
        plt.show()


    



class Receiver:

    def __init__(self):

        pass



    @staticmethod
    def add_receiver_noise(data, snr):

        
        signal_power = np.mean(abs(data**2))
        snr = 10**(snr/10)

        std = np.sqrt(signal_power/snr)
        noise_of_receiver = np.random.normal(0, std, data.shape[0]) + 1j*np.random.normal(0, std, data.shape[0])

        return data + (1/np.sqrt(2))*noise_of_receiver



class Detector:

    def __init__(self):

        pass

    
    def subcarrier_data_estimation(self, data, h, method):
        

        if method == 'zero forcing':
            return (data/h)[subcarrier_carrying_data_index]
        
        if method == 'matched filtering':
            return (np.conj(h)*data)[subcarrier_carrying_data_index]
        
        if method == 'MMSE':

            H = np.conj(h)/(abs(h)**2 + 0.001)
            return (data*H)[subcarrier_carrying_data_index]



    def constellation_vis(self, data, symbols_info):
        plt.figure(facecolor='black')
        plt.plot(np.real(data), np.imag(data), 'r.', label = 'received signal')
        plt.plot(np.real(symbols_info), np.imag(symbols_info), 'y*', label = 'original signal')
        ax = plt.gca()
        ax.set_facecolor('black')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        plt.xlabel('Real', fontsize = 16)
        plt.ylabel('Imag', fontsize = 16)
        ax.tick_params(axis='x', colors='w')
        ax.tick_params(axis='y', colors='w')
        plt.title('   16-QAM     64 sub-carriers', color = 'w')
        plt.xlim(-5,5)
        plt.ylim(-5,5)
        plt.grid(alpha = 0.3)
        plt.legend()
        plt.show()



    def maximum_likelihood_estimator(self, symbols_estimation, symbols_info):

        estimated_value = []
        for idx in range(symbols_estimation.shape[0]):
            I = np.argmin(abs(symbols_estimation[idx] - symbols_info))
            estimated_value.append(symbols_info[I])


        return np.array(estimated_value)





if __name__ == '__main__':

    len_data = 220
    num_subcarriers = 64
    num_pilots = 8
    num_bits_per_symb = 4
    num_cyclic_prefix = int(num_subcarriers/4)
    channel_impulse_response = np.array([1, 1j, 0.1 + 0.5j, 0.2 + 0.2j])
    snr = 30 #db


    subcarrier_index = np.arange(0,num_subcarriers)
    pilot_index = subcarrier_index[::int(num_subcarriers/num_pilots)].tolist()
    pilot_index.append(subcarrier_index[-1])
    num_pilots += 1
    pilot_index = np.array(pilot_index)
    subcarrier_carrying_data_index = np.delete(subcarrier_index, pilot_index)


    pilot_value = 3 + 3j

    utils = Utils()
    channel = Channel()
    detector = Detector()


    bits_of_info_serial = dataGen.DATA_generation(len_data)


    bits_of_info_parallel = utils.Serial_to_Parallel(bits_of_info_serial)

    symbols_info = utils.map_bit_to_symb(bits_of_info_parallel)
    subcarrier_data_pilot_frequency = np.zeros(num_subcarriers, dtype = complex)
    subcarrier_data_pilot_frequency[subcarrier_carrying_data_index] = symbols_info
    subcarrier_data_pilot_frequency[pilot_index] = pilot_value

    subcarrier_data_pilot_time = utils.apply_ifft(subcarrier_data_pilot_frequency)

    subcarrier_data_pilot_time_CP = utils.adding_CP(subcarrier_data_pilot_time, num_cyclic_prefix)

    subcarrier_data_pilot_time_CP_channel = channel.channel_(subcarrier_data_pilot_time_CP, channel_impulse_response)

    subcarrier_data_pilot_time_CP_channel_receiver = Receiver.add_receiver_noise(subcarrier_data_pilot_time_CP_channel, snr)

    subcarrier_data_pilot_time_rx = utils.remove_CP(subcarrier_data_pilot_time_CP_channel_receiver)

    subcarrier_data_pilot_frequency_rx = utils.apply_fft(subcarrier_data_pilot_time_rx)

    channel_impulse_response_est = channel.channel_estimation(subcarrier_data_pilot_frequency_rx)

    channel_impulse_response_est = channel.channel_estimation_interpolation(channel_impulse_response_est)
    
    channel.channel_impulse_response_visualization(channel_impulse_response_est)
    
    symbols_estimation = detector.subcarrier_data_estimation(subcarrier_data_pilot_frequency_rx, channel_impulse_response_est, 'MMSE') #'matched filtering') #'zero forcing')

    detector.constellation_vis(symbols_estimation, symbols_info)

    symbols_estimation = detector.maximum_likelihood_estimator(symbols_estimation, symbols_info)

    #detector.constellation_vis(symbols_estimation, symbols_info)


    bits_estimation = utils.map_symb_to_bit(symbols_estimation).reshape(-1)

    #print(f'Error {abs(bits_estimation - bits_of_info_serial)} ')
