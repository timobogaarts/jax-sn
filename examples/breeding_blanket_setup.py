import numpy as onp
import jax.numpy as jnp
import jax 
import pytest
from functools import partial
jax.config.update("jax_enable_x64", True)

def bb_data_dict():
    data_dict = {
        'BackSupportingStructure': (
            onp.array([14.77209636, 22.96845169, 54.47200191, 55.70180008]),
            onp.array([[[ 1.24942070e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.25795981e+00,  2.28111810e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  9.67748935e-02,  5.36368996e+01,  5.82725123e-03],
            [ 0.00000000e+00,  0.00000000e+00,  1.27187063e-01,  5.36495120e+01]],

            [[ 4.64231618e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-5.90813815e-02,  1.25643926e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -3.10286088e-02,  7.18068214e-01,  4.21791065e-03],
            [ 0.00000000e+00,  0.00000000e+00, -4.20798752e-02,  6.50962964e-01]],

            [[ 3.82012630e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.46871139e-02,  9.61673768e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -8.16188014e-04,  1.28078481e-02,  2.23290762e-03],
            [ 0.00000000e+00,  0.00000000e+00,  1.08101746e-03, -3.08412536e-02]],

            [[ 2.08341971e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.70518233e-02,  7.25544490e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.98970282e-04,  4.42525801e-02,  9.53335424e-04],
            [ 0.00000000e+00,  0.00000000e+00, -2.58888636e-05, -9.47098354e-03]],

            [[ 1.14886241e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-6.08587092e-03,  8.04990888e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.44134581e-04, -9.98122892e-03,  4.12105430e-04],
            [ 0.00000000e+00,  0.00000000e+00, -3.62110923e-04, -1.28478212e-03]],

            [[ 5.59608630e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.94979824e-03,  3.66068753e-03,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  3.15193776e-05,  1.61003934e-02,  1.07907715e-04],
            [ 0.00000000e+00,  0.00000000e+00, -3.32111930e-04, -1.38009632e-02]]])
        ),
        'BreedingZone': (
            onp.array([14.26520099, 33.2374296 , 46.22254283, 64.78551751]),
            onp.array([[[ 1.27263811e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.68263604e+00,  3.22756614e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  2.62800524e-01,  3.99674013e+01,  2.72044739e-03],
            [ 0.00000000e+00,  0.00000000e+00,  1.01872030e-01,  4.01261596e+01]],

            [[ 5.69873316e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.63175461e-02,  2.63653402e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -7.65089532e-02,  2.32328167e+00,  2.60787954e-03],
            [ 0.00000000e+00,  0.00000000e+00, -3.47139625e-02,  2.34997165e+00]],

            [[ 3.56722532e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.21476924e-02,  4.79992444e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -5.16520833e-03,  4.91821817e-02,  2.39643838e-03],
            [ 0.00000000e+00,  0.00000000e+00,  2.52550546e-04,  4.81297066e-03]],

            [[ 2.00462038e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.95537183e-02,  4.83582428e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -8.44775908e-04,  4.30629484e-02,  2.11016838e-03],
            [ 0.00000000e+00,  0.00000000e+00,  4.91736887e-04,  8.59957616e-03]],

            [[ 1.07216613e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-3.41887931e-03,  2.48490116e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -4.63035411e-04,  1.62493885e-03,  1.77783070e-03],
            [ 0.00000000e+00,  0.00000000e+00, -4.66834907e-04, -9.65295073e-02]],

            [[ 4.94932801e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.56340688e-03,  1.77672427e-03,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  6.05788559e-05, -4.16034752e-03,  1.42673026e-03],
            [ 0.00000000e+00,  0.00000000e+00, -4.02825900e-06, -1.24300871e-01]]])
        ),
        'FirstWall': (
            onp.array([15.28360321, 22.77599714, 57.37177834, 59.02067236]),
            onp.array([[[ 1.35739102e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.47264631e+00,  2.27073391e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  3.71920704e-02,  5.67913293e+01,  1.58972770e-02],
            [ 0.00000000e+00,  0.00000000e+00,  2.24933173e-02,  5.73402116e+01]],

            [[ 5.86238206e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 3.42093512e-02,  2.09849877e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.19380091e-02,  6.39940915e-01,  9.59542653e-03],
            [ 0.00000000e+00,  0.00000000e+00, -7.19817190e-03,  6.81404367e-01]],

            [[ 4.70132164e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.32212713e-02,  1.68460566e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  9.36025916e-05,  6.19816549e-02,  2.18780421e-03],
            [ 0.00000000e+00,  0.00000000e+00, -6.53698616e-05, -1.33153790e-01]],

            [[ 3.40430933e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.30475963e-02,  1.62770030e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.23515743e-04,  1.18566666e-01, -2.29875869e-03],
            [ 0.00000000e+00,  0.00000000e+00,  6.21638288e-05,  3.16992612e-02]],

            [[ 2.48736323e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 8.42745972e-04,  1.42515822e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.35970420e-04,  7.19430772e-02, -3.98712470e-03],
            [ 0.00000000e+00,  0.00000000e+00, -1.34316710e-04, -1.73357264e-01]],

            [[ 1.67694811e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.33650490e-04,  3.39274089e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -5.25059140e-04, -7.96544962e-02, -3.85966729e-03],
            [ 0.00000000e+00,  0.00000000e+00,  3.03105854e-06, -6.13323799e-02]]])
        ),
        'Plasma': (
            onp.array([1e-10, 1e-10, 1e-10, 1e-10]),
            onp.array([[[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]],

            [[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]])
        ),
        'TungstenArmor': (
            onp.array([ 35.62796196,  55.19855355, 170.68293579,  97.62430527]),
            onp.array([[[ 3.30981502e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.23786233e+01,  5.39927385e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  3.81105370e-03,  1.44466525e+02,  1.33580218e-01],
            [ 0.00000000e+00,  0.00000000e+00,  6.34121092e-03,  4.26289394e+01]],

            [[ 1.74633605e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 1.42497322e-01,  7.97618562e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -8.29351683e-04,  4.65014502e-01,  3.91297892e-02],
            [ 0.00000000e+00,  0.00000000e+00, -1.87361531e-03,  1.75203903e+00]],

            [[ 1.43568377e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-5.63057553e-02,  3.44580388e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -3.07670565e-04, -7.27429723e-03,  1.95999255e-02],
            [ 0.00000000e+00,  0.00000000e+00, -5.14844961e-04,  3.37708238e-02]],

            [[ 1.19294691e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 7.00295216e-03,  1.13919178e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.43342420e-04, -4.30902562e-01,  4.93136604e-02],
            [ 0.00000000e+00,  0.00000000e+00,  1.46932234e-04,  7.67155835e-01]],

            [[ 9.75011766e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.51936726e-02,  3.19768662e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  2.58680837e-04, -3.37626089e-01,  8.51612036e-03],
            [ 0.00000000e+00,  0.00000000e+00,  5.40664309e-04,  1.22455518e-01]],

            [[ 8.12408076e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-3.15909978e-03,  1.81364732e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  4.02821611e-07,  4.95540772e-01, -3.56710582e-02],
            [ 0.00000000e+00,  0.00000000e+00, -5.33719860e-05,  1.27297386e-01]]])
        ),
        'VVShield': (
            onp.array([ 26.47837879,  82.79717601, 113.04865901, 116.92695037]),
            onp.array([[[ 1.75783052e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 8.75630464e+00,  7.68543671e+01,  0.00000000e+00,  0.00000000e+00],
            [ 8.45647746e-04,  5.53553434e+00,  9.66124369e+01,  1.62193366e-04],
            [ 4.10894125e-04,  1.83052937e-01,  1.55851386e+01,  1.01413898e+02]],

            [[ 7.36928750e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.66212746e+00,  2.54116983e+01,  0.00000000e+00,  0.00000000e+00],
            [ 9.70018107e-05,  2.14171330e+00,  3.00609316e+01,  1.18475218e-04],
            [ 9.56673483e-05,  1.72473635e-02,  6.58129813e+00,  3.19741496e+01]],

            [[ 5.66377710e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 7.93395421e-01,  1.13522342e+01,  0.00000000e+00,  0.00000000e+00],
            [-4.04915178e-04, -6.17607927e-01,  1.51388727e+01,  4.87926363e-05],
            [-1.72036114e-04, -8.69943812e-02, -1.67150354e+00,  1.47688177e+01]],

            [[ 3.01413788e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.10801549e+00,  1.41669720e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.41630228e-04, -1.42752214e+00,  3.99017864e+00, -1.93967678e-05],
            [-1.30536057e-04, -2.43641338e-02, -3.94754939e+00,  3.17768329e+00]],

            [[ 1.57585804e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-9.00345813e-01, -1.18628291e+00,  0.00000000e+00,  0.00000000e+00],
            [ 2.73264081e-04, -4.63214103e-01, -8.43293384e-01, -6.11782273e-05],
            [ 7.58404715e-05,  5.84478464e-02, -1.43987671e+00, -6.47577749e-01]],

            [[ 7.05655289e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.60766923e-01, -4.59181612e-01,  0.00000000e+00,  0.00000000e+00],
            [ 1.68556788e-04,  4.10931592e-01, -1.00158100e+00, -6.48254833e-05],
            [ 1.36212757e-04,  2.82734288e-02,  9.03335911e-01, -9.43603087e-01]]])
        ),
        'VVWallBack': (
            onp.array([25.79247168, 51.11727444, 88.78516302, 92.20411034]),
            onp.array([[[ 2.19699856e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 3.62523764e+00,  5.06345155e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  2.88910130e-01,  8.68210498e+01,  3.11055824e-03],
            [ 0.00000000e+00,  0.00000000e+00,  1.07586836e+00,  8.63905171e+01]],

            [[ 8.17093304e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.45699763e-01,  2.49713044e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.00221140e-01,  1.37514701e+00,  1.84426019e-03],
            [ 0.00000000e+00,  0.00000000e+00, -3.78407230e-01,  7.50659477e-02]],

            [[ 6.64937463e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 6.48673482e-02,  1.66073168e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  6.80970615e-03,  8.67469213e-02,  8.50553181e-05],
            [ 0.00000000e+00,  0.00000000e+00,  8.76649594e-03, -1.42145790e-01]],

            [[ 3.66174097e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-4.63481600e-02,  1.86828389e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -4.45571897e-03,  1.11516247e-01, -1.14519665e-03],
            [ 0.00000000e+00,  0.00000000e+00,  2.43672929e-03,  1.85445612e-01]],

            [[ 2.08281486e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-5.02256373e-03,  1.23637773e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  2.88338906e-03, -1.69530267e-02, -1.25185323e-03],
            [ 0.00000000e+00,  0.00000000e+00,  7.53813912e-04, -2.99178740e-01]],

            [[ 1.01427992e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-1.07524659e-02,  5.78436813e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  2.13188124e-03,  1.57604197e-01, -4.19984418e-04],
            [ 0.00000000e+00,  0.00000000e+00,  1.20025912e-02,  4.25099841e-02]]])
        ),
        'VVWallFront': (
            onp.array([24.44039741, 48.51043365, 88.89183676, 91.29762668]),
            onp.array([[[ 2.02577023e+01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 4.14585073e+00,  4.81968058e+01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.75784854e-01,  8.72837262e+01,  4.72404270e-03],
            [ 0.00000000e+00,  0.00000000e+00,  5.89969998e-01,  8.69088138e+01]],

            [[ 6.76159895e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.61604791e-01,  2.19505396e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -5.59894292e-02,  9.57622837e-01,  3.15556802e-03],
            [ 0.00000000e+00,  0.00000000e+00, -2.02398281e-01,  1.58826704e+00]],

            [[ 5.57742629e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 9.32512768e-02,  1.43047190e+00,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.89497192e-03, -2.55776254e-02,  1.12146368e-03],
            [ 0.00000000e+00,  0.00000000e+00,  3.59768572e-03, -4.71665117e-01]],

            [[ 2.51342530e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-6.25062066e-02,  1.09156981e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00,  1.59441999e-03,  1.65931463e-01, -1.51215029e-04],
            [ 0.00000000e+00,  0.00000000e+00,  4.80587532e-03, -2.15759398e-01]],

            [[ 1.32974418e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [-2.91303001e-02,  1.17182597e-01,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -8.42861088e-04, -8.22885192e-03, -2.96211260e-04],
            [ 0.00000000e+00,  0.00000000e+00, -6.07784044e-03, -1.14983166e-01]],

            [[ 4.99412662e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
            [ 4.13827372e-03,  2.46399333e-02,  0.00000000e+00,  0.00000000e+00],
            [ 0.00000000e+00, -1.53108267e-03, -3.38389056e-02,  1.14031420e-04],
            [ 0.00000000e+00,  0.00000000e+00,  1.61791981e-03,  1.62309618e-02]]])
        ),
    }
    return data_dict

def aux_data_dict():
    aux_data_dict = {
    'BackSupportingStructure': {
        'heating': jnp.array([1.96243487e+04, 7.64483843e+02, 4.57802819e+00, 1.15483421e+01]),
        'nt': jnp.array([2.11062784e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    'BreedingZone': {
        'heating': jnp.array([ 119806.09732053,   40496.41088611,  289594.04437045, 1167898.82782947]),
        'nt': jnp.array([0.00127852, 0.00682984, 0.06053289, 0.24413547]),
    },
    'FirstWall': {
        'heating': jnp.array([6.70264822e+04, 1.54226304e+03, 5.10790572e+00, 8.42094829e+00]),
        'nt': jnp.array([6.84443328e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    'Plasma': {
        'heating': jnp.array([1.37372566e+04, 3.85362758e+02, 4.58094919e-01, 8.15790335e-01]),
        'nt': jnp.array([8.51041773e-13, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    'TungstenArmor': {
        'heating': jnp.array([9.16728376e+03, 1.28632502e+03, 7.33933737e+00, 1.02496732e+01]),
        'nt': jnp.array([6.81821974e-41, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    'VVShield': {
        'heating': jnp.array([1.08898662e+05, 1.59711262e+04, 4.52674365e+01, 1.49537471e+02]),
        'nt': jnp.array([6.59770973e-07, 9.17694249e-12, 4.60891642e-11, 3.88480694e-10]),
    },
    'VVWallBack': {
        'heating': jnp.array([4.18746225e+04, 1.78587444e+03, 3.40032171e+01, 2.16743286e+02]),
        'nt': jnp.array([1.56829474e-06, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    'VVWallFront': {
        'heating': jnp.array([25402.37299295,  1453.44172751,    31.33813679,   175.37812099]),
        'nt': jnp.array([7.38823703e-07, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]),
    },
    }
    return aux_data_dict

def base_region_sizes():
    region_sizes = [0.2, 0.002, 0.025, 0.5, 0.385, 0.06, 0.2, 0.06]
    return region_sizes

def base_resolutions():
    return [5,1,1,5,5,1,5,2]

def _interp_x():
    return onp.linspace(0 + 1e-5, onp.sum(base_region_sizes()) - 1e-5, 1000)

def _bb_sb(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order):    
    import StellBlanket as SB
    from .test_anisotropic_scattering import _switch_element_type

    xs_data_dict = bb_data_dict()
    
    region_sizes = base_region_sizes()

    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone','BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    xs_regions   = [SB.Generic_Elements.Material(SB.Generic_Elements.Multigroup_Cross_Section(xs_data_dict[i][0], xs_data_dict[i][1]) ) for i in region_names]

    quad_set     = SB.Generic_Elements.Quadrature_Set(*SB.Generic_Elements.create_TN(tn_order))
    element      = _switch_element_type(dim)
    energy_set   = SB.Generic_Elements.Energy_Set(onp.arange(xs_data_dict['Plasma'][0].shape[0] + 1))    
    source_p = SB.Generic_Elements.Constant_Source(element, 1.0, energy_set[0])
    source_0 = SB.Generic_Elements.Zero_Source(element)
    
    sources = [source_p if i ==0 else  source_0 for i in range(len(xs_regions))]
    lmax = xs_data_dict['Plasma'][1].shape[0] -1
    cartesian_domain, source = SB.Generic_Elements.Create_Slab(element, region_sizes, n_elem_regions, xs_regions, d_other_dim, n_other_dim, sources)            
    matfreente       = SB.Generic_Elements.Create_RAM_Simulation(quad_set, energy_set, cartesian_domain, degree, lmax)
    return matfreente, source





def setup_bb(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order):

    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex    
    from jax_sn.quadrature_set import create_tn_quadrature_set

    data_dictionary = bb_data_dict() # dictionary with matrices

    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering = jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}

    
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)

    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone','BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']

    xs_regions = [xs_data_dict[i] for i in region_names]


    region_size_res = [(size, n_elem) for size, n_elem in zip(d_elem_regions, n_elem_regions)]

    
    source_eg0 = [1 if i ==0 else 0 for i in range(len(region_names))]
    

    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res] + other_dims, xs_regions, source_eg0, 0)
    element_type                         = BasixLagrangianSimplex(degree, dim)
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    # isotropic flat source
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    return solution_domain, sources, create_tn_quadrature_set(tn_order)

def _setup_bb_args(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order):
    from jax_sn.operators.scatter_operator import GatherScatterOperatorBroadcasted
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, VmapForward, VmapSweep, VmapAngles
    from jax_sn.operators.mass_operator import MassOperator
    from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
    from jax_sn.operators.transport_operator import BasicSweepData
    from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan
    

    solution_domain, sources, quadrature_set = setup_bb(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order) 

    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    sweep_data = BasicSweepData.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAngles(TransportOperator(solution_domain.solution_domain_data, sweep_data, 0, VmapForward(), VmapSweep()), VmapAngles())
    mass_operator             = MassOperator.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)    
    
    groups_elements_basis_shape = (n_energy_groups, ) + solution_domain.elements_basis_shape.shape 


    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = GatherScatterOperatorBroadcasted(cross_sections_indexed=solution_domain.domain.cross_sections_indexed)
    source_total              = jnp.broadcast_to(sources[None, :, :], (quadrature_set.angles.shape[0],) + sources.shape) # source for all angles
    Aop                       = MultiGroupOperatorScan(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, n_energy_groups)
    return Aop, source_total, solution_domain

def _get_bb_args(dim, degree):
    d_elem_regions = base_region_sizes()
    n_elem_regions = base_resolutions()
    d_other_dim    = 5.0
    n_other_dim    = 1
    tn_order       = 3
    return dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order

def _get_solution_interp(solution_domain, value):
    x = jnp.linspace(0, jnp.max(solution_domain.solution_domain_data.domain.vertices[:,0]), 100)[:, None]
    solution_values = jnp.array([solution_domain.interpolate(x, value[i,0,...]) for i in range(value.shape[0])])
    return x, solution_values

def plot_solution_values(solution_domain, value, ):
    import matplotlib.pyplot as plt
    x, solution_values = _get_solution_interp(solution_domain, value)
    plt.semilogy(x, solution_values.T)
    plt.xlabel("x")
    plt.ylabel(r"$\phi$")
    


def setup_bb_symmetric(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order, data_dictionary = None):

    def _symmetrize_no_plasma(list):
        return [list[-( i + 1)] for  i in range(len(list) - 1)] + list

    from jax_sn.domain.geometric_domain import CrossSectionData
    from jax_sn.domain.domain_generation import create_cartesian_setup
    from jax_sn.domain import Domain
    from jax_sn.solution_domain import SolutionDomain, BasixLagrangianSimplex    
    from jax_sn.quadrature_set import create_tn_quadrature_set
    if data_dictionary is None:
        data_dictionary = bb_data_dict() # dictionary with matrices

    xs_data_dict = {name : CrossSectionData(total = jnp.array(data_dictionary[name][0]), scattering = jnp.array(data_dictionary[name][1])) for name in data_dictionary.keys()}
    
    other_dims   = [[(d_other_dim, n_other_dim)]]*(dim - 1)

    region_names = ['Plasma', 'TungstenArmor', 'FirstWall', 'BreedingZone','BackSupportingStructure', 'VVWallFront', 'VVShield', 'VVWallBack']
    region_size_res = [(size, n_elem) for size, n_elem in zip(d_elem_regions, n_elem_regions)]

    region_size_res_symm = _symmetrize_no_plasma(region_size_res)    
    source_eg0_symm      = _symmetrize_no_plasma([1 if i ==0 else 0 for i in range(len(region_names))])
    xs_regions_symm      = _symmetrize_no_plasma([xs_data_dict[name] for name in region_names])
    
    verts, conn, cross_sections, sources = create_cartesian_setup(dim, [region_size_res_symm] + other_dims, xs_regions_symm, source_eg0_symm, 0)    
    element_type                         = BasixLagrangianSimplex(degree, dim)
    domain                               = Domain.from_mesh_and_cross_sections((verts, conn), element_type.face_template, cross_sections)    
    solution_domain                      = SolutionDomain.from_element_and_domain(element_type, domain)
    # isotropic flat source
    sources = jnp.repeat(sources, repeats = solution_domain.n_basis, axis = -1).reshape(-1, solution_domain.n_basis)

    return solution_domain, sources, create_tn_quadrature_set(tn_order)

def _setup_bb_mgop_symmetric(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order, data_dictionary = None):
    from jax_sn.operators.scatter_operator import GatherScatterOperatorBroadcasted
    from jax_sn.operators.transport_operator import TransportOperator, TransportOperatorMultipleAngles, VmapForward, VmapSweep, VmapAngles
    from jax_sn.operators.mass_operator import MassOperator
    from jax_sn.operators.moment_operator import MomentOperator, DiscreteOperator
    from jax_sn.operators.transport_operator import BasicSweepData
    from jax_sn.operators.multi_group_operator import MultiGroupOperatorScan
    
    

    solution_domain, sources, quadrature_set = setup_bb_symmetric(dim, degree, d_elem_regions, n_elem_regions, d_other_dim, n_other_dim, tn_order, data_dictionary) 

    n_energy_groups           = solution_domain.domain.cross_sections_indexed.n_energy_groups
    lmax                      = solution_domain.domain.cross_sections_indexed.l_max

    sweep_data = BasicSweepData.from_domain_quadrature_set(solution_domain.domain, quadrature_set, n_iterations_per_loop = 100)
    transport_operator        = TransportOperatorMultipleAngles(TransportOperator(solution_domain.solution_domain_data, sweep_data, 0, VmapForward(), VmapSweep()), VmapAngles())
    mass_operator             = MassOperator.from_transport_operator(transport_operator)
    moment_operator           = MomentOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)    
    
    groups_elements_basis_shape = (n_energy_groups, ) + solution_domain.elements_basis_shape.shape 


    discrete_operator         = DiscreteOperator.from_quadrature_set(quadrature_set, l_max = lmax, elements_basis_shape=solution_domain.elements_basis_shape)
    gather_scatter_operator   = GatherScatterOperatorBroadcasted(cross_sections_indexed=solution_domain.domain.cross_sections_indexed)
    source_total              = jnp.broadcast_to(sources[None, :, :], (quadrature_set.angles.shape[0],) + sources.shape) # source for all angles
    Aop                       = MultiGroupOperatorScan(moment_operator, discrete_operator, transport_operator, mass_operator, gather_scatter_operator, n_energy_groups)
    return Aop, source_total, solution_domain


@partial(jax.jit, static_argnums = 1)
def recreate_symmetric_vertices(size_layers, resolution_layers):
    n_regions_dim= size_layers.shape[0]
    cumsum_nodes  = jnp.concatenate([jnp.array([0]), jnp.cumsum(size_layers)])
    plasma_to_blanket_edge =  jnp.concatenate([jnp.linspace(cumsum_nodes[i], cumsum_nodes[i+1], resolution_layers[i], endpoint=False) for i in range(n_regions_dim)] + [jnp.array([cumsum_nodes[-1]])])
    blanket_size = cumsum_nodes[-1] - cumsum_nodes[1] 
    return jnp.concatenate([plasma_to_blanket_edge[-1] - jnp.flip(plasma_to_blanket_edge)[:-resolution_layers[0]], plasma_to_blanket_edge[1:] + blanket_size], axis=0)


import mcbb 
from typing import Tuple, Iterable
from functools import lru_cache


def create_openmc_data(blanket : mcbb.linear_blanket.Blanket1D, groups_list : list, settings : mcbb.blanket_base.OpenMCSettingsBlanket, l_order, n_x_tally = 200) -> Tuple[dict, dict, dict]:
    
    simulation_symm = mcbb.linear_blanket.OpenMCSymmetricLinearBlanketSimulation(blanket, settings, n_x_tally, groups_list)
    mgxs_lib, tally_result_dict =simulation_symm.create_mgxs_library(onp.flip(groups_list), l_order)
    mgxs_lib_new = mcbb.openmc_base_functions.mgxs_lib_to_standard_formatting(mgxs_lib)
    new_bb_data_dict = {key : (mgxs_lib_new[key]['total'], mgxs_lib_new[key]['nu-scatter matrix']) for key in bb_data_dict().keys()}
    new_aux_data_dict = {key : {'nt' : mgxs_lib_new[key]['(n,Xt)']} for key in aux_data_dict().keys()}
    return new_bb_data_dict, new_aux_data_dict, tally_result_dict

@lru_cache(maxsize=  4)
def _get_hcpb_data_openmc(d_elem_regions : Iterable[float], l_order : int, n_samples_per_batch : int, groups_list : Iterable[float]) -> Tuple[dict, dict, dict]:
    # Returns bb_data_dict, aux_data_dict, tally_result_dict
    hcpb_blanket = mcbb.examples.hcpb()
    for i, layer in enumerate(hcpb_blanket.layers):
        if i == 0:
            layer.thickness = 0.5 * d_elem_regions[i]
        else:
            layer.thickness = d_elem_regions[i]
    settings = mcbb.blanket_base.OpenMCSettingsBlanket(True, "openmc_data/", 10, n_samples_per_batch)
    return create_openmc_data(hcpb_blanket, groups_list, settings, l_order)
