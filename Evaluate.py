import PIL.Image as Image
import numpy as np
import PIL.Image as Image
import numpy as np
from Evaluator import image_read_cv2, Evaluator
import os

# ir_path = 'data/SPECT-MRI/test/SPECT'
# vis_path = 'data/SPECT-MRI/test/MRI'
# output_dir = 'output/SPECT-MRI/MRCENet_lgradint_latest'

ir_path = 'data/CT-MRI/test/CT'
vis_path = 'data/CT-MRI/test/MRI'
output_dir = 'output/CT-MRI/MRCENet_lgradint_latest'

# ir_path = 'data/PET-MRI/test/PET'
# vis_path = 'data/PET-MRI/test/MRI'
# output_dir = 'output/PET-MRI/MRCENet_lgradint_latest' 

# ir_path = 'data/Whu_Data_82/test/PET'
# vis_path = 'data/Whu_Data_82/test/MRI'
# output_dir = 'output/Whu_Data_82/Net_v25_11010_n9'

files = os.listdir(output_dir)

metric_result = np.zeros((11))

for file in files:
    ir = image_read_cv2(ir_path + '/' + file, 'GRAY')
    vis = image_read_cv2(vis_path + '/' + file,'GRAY' )
    fi = image_read_cv2(output_dir + '/' + file, 'GRAY')

    metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                  , Evaluator.SF(fi), Evaluator.MI(fi, ir, vis)
                                  , Evaluator.SCD(fi, ir, vis), Evaluator.VIFF(fi, ir, vis)
                                  , Evaluator.Qabf(fi, ir, vis), Evaluator.SSIM(fi, ir, vis), Evaluator.PSNR(fi, ir, vis)
                                  , Evaluator.AG(fi), Evaluator.MSE(fi, ir, vis)])
metric_result /= len(files)
model_name = output_dir.split('/')[-1]

print(model_name+'\t'+ 'EN:' + str(np.round(metric_result[0], 4))+'\t'
                + 'SD:' + str(np.round(metric_result[1], 4))+'\t'
                + 'SF:' + str(np.round(metric_result[2], 4))+'\t'
                + 'MI:' + str(np.round(metric_result[3], 4))+'\t'
                + 'SCD:' + str(np.round(metric_result[4], 4))+'\t'
                + 'VIFF:' + str(np.round(metric_result[5], 4))+'\t'
                + 'Qabf:' + str(np.round(metric_result[6], 4))+'\t'
                + 'SSIM:' + str(np.round(metric_result[7], 4))+'\t'
                + 'PSNR:' + str(np.round(metric_result[8], 4))+'\t'
                + 'AG:' + str(np.round(metric_result[9], 4))+'\t'
                + 'MSE:' + str(np.round(metric_result[10], 4))+'\t'
                #+ 'CC:' + str(np.round(metric_result[10], 4))+'\t'
                )
print("="*195)

