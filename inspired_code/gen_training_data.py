# %%
import numpy as np
import h5py
import time

class IQSampleLoader:
    def __init__(self, dataset_name, labelset_name):
        self.dataset_name = dataset_name
        self.labelset_name = labelset_name

    def _convert_to_complex(self, data):
        '''Convert the loaded data to complex IQ samples.'''
        num_row = data.shape[0]
        num_col = data.shape[1]
        data_complex = np.zeros([num_row, round(num_col/2)], dtype=np.complex64)
        data_complex = data[:, :round(num_col/2)] + 1j * data[:, round(num_col/2):]
        return data_complex

    def load_iq_samples(self, file_path, dev_range, pkt_range):
        '''
        Load IQ samples from a dataset.

        INPUT:
            file_path: The dataset path.
            dev_range: Specifies the loaded device range.
            pkt_range: Specifies the loaded packets range.

        RETURN:
            data: The loaded complex IQ samples.
            label: The true label of each received packet.
        '''

        f = h5py.File(file_path, 'r')
        label = f[self.labelset_name][:]
        label = label.astype(int)
        label = np.transpose(label)
        label = label - 1

        label_start = int(label[0][0]) + 1
        label_end = int(label[-1][0]) + 1
        num_dev = label_end - label_start + 1
        num_pkt = len(label)
        num_pkt_per_dev = int(num_pkt / num_dev)

        print('Dataset information: Dev ' + str(label_start) + ' to Dev ' +
              str(label_end) + ', ' + str(num_pkt_per_dev) + ' packets per device.')

        sample_index_list = []

        for dev_idx in dev_range:
            sample_index_dev = np.where(label == dev_idx)[0][pkt_range].tolist()
            sample_index_list.extend(sample_index_dev)
            # print(sample_index_dev)
        data = f[self.dataset_name][sample_index_list]
        data = self._convert_to_complex(data)

        label = label[sample_index_list]

        f.close()
        return data, label

class ChannelIndSpectrogram():
    def __init__(self):
        pass

    def _normalization(self, data):
        '''Normalize the signal.'''
        s_norm = np.zeros(data.shape, dtype=complex)

        for i in range(data.shape[0]):
            sig_amplitude = np.abs(data[i])
            rms = np.sqrt(np.mean(sig_amplitude**2))
            s_norm[i] = data[i] / rms

        return s_norm

    def _gen_single_channel_CAF(self, sig):
    
        # Finding the right alpha
        CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
        for j in range(len(alphas)):
            for i in range(len(taus)):
                CAF[j, i] = np.sum(sig *
                            np.conj(np.roll(sig, taus[i])) *
                            pre_exp[j])
                
        #CAF2=CAF.copy()
        #CAF2[60] = 0

        return np.abs(CAF)

    def channel_ind_CAF(self, data):
        '''Converts the data to channel-independent spectrograms.'''
        data = self._normalization(data)
        num_sample = data.shape[0]
        num_row = len(alphas)  
        num_column = len(taus)
        data_channel_ind_spec = np.zeros([num_sample, num_row, num_column, 1])

        for i in range(num_sample):
            chan_ind_spec_amp = self._gen_single_channel_CAF(data[i])
            data_channel_ind_spec[i, :, :, 0] = chan_ind_spec_amp

        return data_channel_ind_spec


# %%
# Usage example
if __name__ == "__main__":
    rel_path = "./image_generation/"

    file_path = rel_path+'data/dataset_training_aug.h5'
    dev_range = np.arange(0, 20, dtype=int)
    pkt_range = np.arange(0, 1000, dtype=int)
    print("generate IQ data")
    LoadDatasetObj = IQSampleLoader(dataset_name='data', labelset_name='label')
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)

    # %%

    taus = np.arange(0, 600)
    alphas = np.arange(-0.3, 0.3, 0.005)
    N=len(data[0])
    pre_exp = np.zeros(N)
    pre_exp = [np.exp(-2j * np.pi * alphas[j] * np.arange(N)) for j in range(len(alphas))]

    print("generate training data")
    ch = ChannelIndSpectrogram()
    t1=time.time()
    data_channel_ind_spec = ch.channel_ind_CAF(data)
    t2=time.time()

    #%%
    diff=t2-t1
    print("la génération a dure "+str(round(diff))+" secondes")

    #%%
    data_channel_ind_spec.dump(rel_path+"output_dat/data_imgs_caf_abs_.dat")
    label.dump(rel_path+"output_dat/label2.dat")

# %%
