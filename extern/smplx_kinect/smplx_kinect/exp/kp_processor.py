import cv2
import numpy as np


def get_kp_processors(args):
    result = {
        k: KPProcessor(v)
        for k, v in args.items()
    }
    return result


class KPProcessor:
    def __init__(self, args):
        self.substract_pelvis = args.get('substract_pelvis', False)
        self.azure_rotate = args.get('azure_rotate', None)
        if self.azure_rotate is not None:
            for sn in self.azure_rotate:
                self.azure_rotate[sn] = np.array(self.azure_rotate[sn])

    def process(self, kp, meta=None):
        if self.substract_pelvis:
            kp = kp - kp[[0]]
        if self.azure_rotate is not None and meta is not None:
            if meta['ds'] == 'azure':
                sn = meta['fix_sample_info'].sn
                if sn in self.azure_rotate:
                    aa = self.azure_rotate[sn]
                    rotmtx = cv2.Rodrigues(aa)[0]
                    kp = kp @ rotmtx.T
        return kp
