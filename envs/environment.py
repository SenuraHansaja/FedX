import math
from numpy import random
import numpy as np
from utils.setting_setup import *
import scipy
from dataset_est.get_entropy import *
from envs.env_utils import *
from envs.env_agent_utils import *
from envs.commcal_utils import *
from utils.func_utils import *
from dataset_est.mnist_est import * ## importing the kl_div_mnist function
from dataset_est.cifar_est import *


class SCFL_env(env_utils, env_agent_utils):
    def __init__(self, args):
        # Network setting
        self.num_Iglob = None
        self.sigma_data = 0.01
        self.lamda = convert_mhz_to_m(args.freq_carrier)
        self.freq_carrier = args.freq_carrier * (10 ** 6)
        self.N_User = args.user_num
        self.G_CU_list = np.ones((self.N_User, 1))  # User directivity
        self.G_BS_t = 1  # BS directivity
        self.Num_BS = 1  # Number of Base Stations
        self.max_step = args.max_step
        self.Z_u = 10000  # Data size

        # Power setting
        self.p_u_max = dBm2W(args.poweru_max)
        self.eta = 0.7  # de tinh R_u
        self.N0 = 3.9811 * (10 ** (-21))  # -174 dBm/Hz -> W/Hz
        # Bandwidth
        self.B = args.bandwidth

        # Base station initialization
        self.BS_x = 0
        self.BS_y = 0
        self.BS_R_Range = 0.1
        self.BS_R_min = 0.01

        # initialization
        self.low_freq = args.low_freq
        self.high_freq = args.high_freq
        self.C_u = np.random.uniform(low=self.low_freq, high=self.high_freq, size=self.N_User)
        self.D_u = 50
        
        self.pen_coeff = args.pen_coeff  # coefficient of penalty defined by lamda in paper
        self.data_size = args.data_size
        # effective switched capacitance that depends on the chip architecture
        self.kappa = 10 ** (-28)
        self.f_u_max = args.f_u_max
        self.skip_max = args.skip_max

        self.xi = 0.5
        self.Time_max = args.tmax  # max time per round
        self.sample_delay = args.sample_delay
        self.sample_skip = 1
        self.S_coeff = args.sample_coeff
        

        # AI Model/Dataset Coefficient
        self.gamma = args.L * (1 - uniform_generator(mean=0, std=0.1))
        self.Lipschitz = args.L
        self.delta = (2 / args.L) * (1 - uniform_generator(mean=0.2, std=0.2))
        self.local_acc = args.local_acc  # target local accuracy
        self.target_acc = args.global_acc  # target global accuracy
        
        
        """Gen Gap caluculation depending on the sampling ratio"""
        self.sampling_ratio = random.random()
        

        """ Generalization Gap Calculation """
        self.dataset = "mnist"  # Choose the dataset to get the entropy value , option ["mnist","cifar10"]
        if self.dataset == "mnist":
            self.entropyH = 3.3  # entropy_holder.get_value("mnist_data")

            print("Value entropy of MNIST dataset: ", self.entropyH)
            
            self.Psi = (1/(2*self.entropyH)) * np.sqrt(2* kl_div_mnist(self.sampling_ratio)) ## based on the new version of the FedX paper
        elif self.dataset == "cifar10":
            self.entropyH = 3.3  # entropy_holder.get_value("cifar10_dataset")
        
            print("Value entropy of CIFAR10 dataset", self.entropyH)
            
            self.Psi = (1/(2 ** self.entropyH)) * np.sqrt(2* kl_div_mnist(self.sampling_ratio))   ## based on the new version of the FedX paper
        else:
            print("Invalid key")
            
    
        print(f"Initial Psi :{self.Psi}")

        """ =============== """
        """     Actions     """
        """ =============== """
        self.beta = np.random.randint(0, self.N_User, size=[self.N_User, 1])  ## bandwidth part
        self.beta = scipy.special.softmax(self.beta, axis=None)  ## bandwidth part
        self.f_u = np.reshape((np.random.rand(1, self.N_User) * self.f_u_max), (self.N_User, 1))
        
        self.sampling_ratio = random.random() ### newly added for the FedX papers
         
        self.p_u = np.reshape((np.random.rand(1, self.N_User) * self.p_u_max), (self.N_User, 1))  ### add code for the ratio of the number of samples  
        self.butt = np.reshape((np.random.rand(1, 1) * self.p_u_max), (1, 1))
        self.tau = np.reshape((np.random.rand(1, 1) * self.p_u_max), (1, 1))

        """ ========================================= """
        """ ===== Function-based Initialization ===== """
        """ ========================================= """
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        self.distance_CU_BS = self._distance_Calculated(self.U_location, self.BS_location)

        self.ChannelGain = self._ChannelGain_Calculate(self.sigma_data)
        self.commonDataRate = self._calculateDataRate(self.ChannelGain)
        self.E = 0  # initialize rewards

        """ ============================ """
        """     Environment Settings     """
        """ ============================ """
        self.rewardMatrix = np.array([])
        self.observation_space = self._wrapState().squeeze()
        self.action_space = self._wrapAction()

    def step(self, action, step):
        penalty = 0
        self.beta, self.f_u, self.p_u, self.butt, self.sampling_ratio = self._decomposeAction(action)  #
        # self.beta, self.f_u, self.p_u = self._decomposeAction(action)  #

        # print(f"beta: {self.beta}")
        # print(f"f_u: {self.f_u}")
        # print(f"p_u: {self.p_u}")
        # print(f"butt: {self.butt}")
        # print(f"sample_delay: {self.sample_delay} * {self.sample_skip} = {self.sample_delay*self.sample_skip}")
        # Environment change
        self.User_trajectory = np.expand_dims(self._trajectory_U_Generator(), axis=0)
        self.U_location = self.User_trajectory + self.U_location
        state_next = self._wrapState()  # State wrap
        self.ChannelGain = self._ChannelGain_Calculate(self.sigma_data)  # Re-calculate channel gain

        self.E = self._Energy()  # Energy
        """============= Global Iter ============="""
        self.num_Iglob = self._calculateGlobalIteration()  # Global Iterations
        if self.num_Iglob < 0:
            print(f"local_acc:{self.butt}|skip:{self.sample_skip}")
        self.Au = self.factor_Iu * self.C_u * self.D_u  # Iterations x Cycles x Samples
        """Penalty"""
        penalty += max(np.sum((self.Au / self.f_u + self.t_trans) - self.Time_max), 0)
        # penalty += max(np.sum(self.t_trans - self.Time_max), 0)
        self.penalty = penalty
        """Minimize E / Minimize num_Iglob / Minimize penalty"""
        
        
        """the new reward function in the FedX paper has both the Energy and Time cumulative reward
            - Mglob in FedX is same as the self.num_Iglob  in the SCFL
            - normalise energy function  - E/E_max -- E_max here is 
            - and the Time factor is T/T_max -- T_max here is 0.5
            new reward function is 
            reward = Mglob(beta* (E_glob/E_max)+ (1-beta)(T_glob/T_max))
            T has to be calculated via the Datarate code
        """
        beta = 0.5

        reward = self.num_Iglob * (beta * (self.E / 0.5) - (1-beta)* ( (self.data_size / self.commonDataRate) / 0.5 ))
        # reward = - np.average(self.t_trans)
        # Stop at Maximum Glob round
        if step == self.max_step:  # or (step == self.num_Iglob):
            done = True
        else:
            done = False
        info = None
        return state_next, reward, done, info

    def reset(self):
        #  System initialization
        self.BS_location = np.expand_dims(self._location_BS_Generator(), axis=0)
        self.U_location = self._location_CU_Generator()
        self.User_trajectory = self._trajectory_U_Generator()
        # Distance calculation
        self.distance_CU_BS = self._distance_Calculated(self.BS_location, self.U_location)

        # re-calculate channel gain
        self.ChannelGain = self._ChannelGain_Calculate(self.sigma_data)
        state_next = self._wrapState()
        return state_next

    def set_attribute(self, key, val):
        if hasattr(self, key):
            setattr(self, key, val)
        else:
            print(f"{key} is not an attribute of the class.")


if __name__ == '__main__':
    args = get_arguments()
    env = SCFL_env(args)
