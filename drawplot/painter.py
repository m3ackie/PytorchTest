import sklearn.metrics as skl
import matplotlib as plt
import matplotlib.pylab as lab


class painter():
    def __init__(self, patient_num, label, prediction, score):
        self.patient_num = patient_num
        self.label = label
        self.prediction = prediction
        self.confusion_matrix = skl.confusion_matrix(label, prediction)
        self.score = score
        self.TN = self.confusion_matrix[0][0]
        self.TP = self.confusion_matrix[1][1]
        self.FN = self.confusion_matrix[1][0]
        self.FP = self.confusion_matrix[0][1]
        self.neg = self.confusion_matrix[0][0] + self.confusion_matrix[0][1]
        self.pos = self.confusion_matrix[1][0] + self.confusion_matrix[1][1]

    def FROC(self):
        fpr, tpr, thresholds = skl.roc_curve(self.label, self.score, pos_label=1)
        sensitivity = tpr
        fps = fpr * self.neg / self.patient_num

        lab.plot(fps, sensitivity, color='b', lw=2)
        lab.legend(loc='lower right')
        lab.xlim([0, 50])
        lab.ylim([0, 1.1])
        lab.xlabel('Average number of false positives per patient')  # 横坐标是fpr
        lab.ylabel('True Positive Rate')  # 纵坐标是tpr
        lab.title('FROC performence')
        lab.show()


if __name__ == '__main__':
    patient_num = 50
    label = [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0,
 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1,
 1, 1, 1, 1, 0, 0, 0, 1, 1]
    prediction =[0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0,
 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ,0 ,0 ,0 ,1 ,1 ,0 ,0 ,0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
 0, 0, 0 ,0, 0, 0, 1, 0, 1]
    score = [0.4971, 0.5375, 0.0343, 0.7269, 0.2563, 0.1393, 0.2607, 0.6736, 0.5846,
        0.2684, 0.2276, 0.3176, 0.1516, 0.4008, 0.3190, 0.0418, 0.5334, 0.6537,
        0.2323, 0.1816, 0.2964, 0.4533, 0.0608, 0.2444, 0.0120, 0.2098, 0.2554,
        0.5124, 0.3010, 0.1087, 0.4779, 0.1326, 0.0402, 0.6628, 0.1535, 0.5463,
        0.3550, 0.0490, 0.2008, 0.2574, 0.0991, 0.2046, 0.3745, 0.1359, 0.2079,
        0.6121, 0.0354, 0.1729, 0.1659, 0.0472, 0.1100, 0.1069, 0.6051, 0.0513,
        0.0624, 0.3540, 0.5575, 0.1254, 0.3977, 0.5446, 0.1211, 0.2577, 0.7344,
        0.2831, 0.4076, 0.1232, 0.5449, 0.0125, 0.0702, 0.0731, 0.5605, 0.5447,
        0.0917, 0.5167, 0.4652, 0.0212, 0.3472, 0.1212, 0.5182, 0.1013, 0.7046,
        0.1444, 0.0160, 0.1631, 0.4471, 0.4452, 0.0546, 0.2237, 0.3070, 0.0709,
        0.1147, 0.0540, 0.6835, 0.5296, 0.0747, 0.0221, 0.0390, 0.0470, 0.1575,
        0.0568, 0.0088, 0.2651, 0.2608, 0.0441, 0.5784, 0.3799, 0.1538, 0.2074,
        0.0710, 0.5295, 0.2145, 0.4490, 0.0948, 0.0746, 0.0340, 0.5266, 0.1263,
        0.5595, 0.1832, 0.3139, 0.4558, 0.1143, 0.0247, 0.4490, 0.1724, 0.4479,
        0.1487, 0.5034, 0.0617, 0.4332, 0.0119, 0.3874, 0.1168, 0.1082, 0.0971,
        0.0521, 0.5059, 0.3433, 0.4888, 0.4781, 0.0956, 0.2059, 0.0480, 0.2583,
        0.3086, 0.0402, 0.3924, 0.0427, 0.0318, 0.1702, 0.4819, 0.0499, 0.0519,
        0.3035, 0.5089, 0.3323, 0.5455]
    print(len(prediction))
    a = painter(patient_num, label, prediction, score)
    a.FROC()