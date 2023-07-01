
import argparse
# 定义命令行参数
import os
import datetime
import time
from multiprocessing import Process
import SimpleITK
import numpy as np
from SimpleITK import ReadImage, GetArrayFromImage, GetImageFromArray, WriteImage
# import SimpleITK as sitk
import model

class orderline():
    def __init__(self):

        parser = argparse.ArgumentParser(description='A simple command line tool.')
        parser.add_argument('--file_path', type=str, help='path to open file.',default=r'D:\liyimengPJ\liyimengPJ1\formal\qtTest\C0')
        parser.add_argument('--keep_header', help="don't adjust header", nargs='?', default='Flase', const="True")
        parser.add_argument('--type', type=str, help="specify MRI Image Type", nargs='?',
                                    choices=["C0", "T2", "LGE"],
                                    default="C0")
        parser.add_argument('--save_path', type=str, help="specify save path", nargs='?', default="")
        parser.add_argument('--batch_process', help="process files in a folder", nargs='?', default='True',
                                    const="True")
        args = parser.parse_args()
        self.Batch_processing(args)
    def pre_data(self, args, nii_path, Save_path):
        if args.keep_header == "True":
            self.keep_header = True
        else:
            self.keep_header = False
        self.Save_path = Save_path
        self.nii_path = nii_path
        self.name = args.type
        self.batch_process = args.batch_process
        self.image = np.zeros((1, 1, 1))
        self.adjust_img = np.zeros((1, 1, 1))
        self.direct = 1
        self.adjusted = False
        self.isOpen = False
        self.predicted = False
        self.predic_tion = ""
        self.saveName = "tfrecord"

    def openfiles(self):

        if self.nii_path == "":
            return
        if self.name == "":
            if "T2" in self.nii_path.split("/")[-1]:
                    self.name = "T2"
            if "LGE" in self.nii_path.split("/")[-1]:
                    self.name = "LGE"
            if "C0" in self.nii_path.split("/")[-1]:
                    self.name = "C0"
            if self.name == "":
                    self.name = "C0"

        self.OpenPath = self.nii_path
        self.img = ReadImage(self.OpenPath)
        self.data = GetArrayFromImage(self.img)
        self.data = np.array(self.data).transpose(1, 2, 0)
        self.spacing = self.img.GetSpacing()
        self.origin = self.img.GetOrigin()
        self.Direction = self.img.GetDirection()
        self.isOpen = True
        self.adjusted = False
        self.predicted = False

    def predict(self):
        try:
            if self.isOpen:
                modul = model.utilss()
                self.predic_tion = modul.predict(self.nii_path)
                print(self.nii_path,':',self.predic_tion)
                self.predicted = True
                self.adjusted = False
            else:
                print("Please input file first")
        except:
            pass

    def adjusting(self):
        DIrection = self.img.GetDirection()
        direction_3x3 = np.reshape(np.array(list(DIrection)), (3, 3))
        direction_3x1x3 = np.expand_dims(direction_3x3, axis=(1))
        a = np.zeros((3, 1, 3))
        a[0] = direction_3x1x3[0:1, :, :]
        a[1] = direction_3x1x3[1:2, :, :]
        a[2] = direction_3x1x3[2:, :, :]
        self.DIrection = np.array(a)

        if self.predic_tion == "000":
            self.DIrection = self.DIrection
        if self.predic_tion == "001":
            self.DIrection[0] = -self.DIrection[0]
        if self.predic_tion == "010":
            self.DIrection[1] = -self.DIrection[1]
        if self.predic_tion == "011":
            self.DIrection[:2, :] = -self.DIrection[:2, :]
        if self.predic_tion == "100":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
        if self.predic_tion == "101":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[0] = -self.DIrection[0]
        if self.predic_tion == "110":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[1] = -self.DIrection[1]
        if self.predic_tion == "111":
            self.DIrection[[0, 1]] = self.DIrection[[1, 0]]
            self.DIrection[:2, :] = -self.DIrection[:2, :]
        self.DIrection = tuple(np.reshape(self.DIrection, (9,)).tolist())
        return True

    def adjust(self):
        if self.predicted:
            self.data = GetArrayFromImage(self.img)
            self.data = np.array(self.data).transpose(1, 2, 0)
            if self.predic_tion == "000":
                self.data = self.data
            if self.predic_tion == "001":
                self.data = np.fliplr(self.data)
            if self.predic_tion == "010":
                self.data = np.flipud(self.data)
            if self.predic_tion == "011":
                self.data = np.flipud(np.fliplr(self.data))
            if self.predic_tion == "100":
                self.data = self.data.transpose((1, 0, 2))
            if self.predic_tion == "101":
                self.data = np.flipud(self.data.transpose((1, 0, 2)))
            if self.predic_tion == "110":
                self.data = np.fliplr(self.data.transpose((1, 0, 2)))
            if self.predic_tion == "111":
                self.data = np.flipud(np.fliplr(self.data.transpose((1, 0, 2))))
            self.direction = '000'
            self.adjusting()
            self.adjusted = True
            return True
        else:
            print("Please input file first")

    def Batch_processing(self,args):
        if args.save_path == "":
            if args.batch_process == "True":
                parent_folder = os.path.dirname(args.file_path)

                folder_name = os.path.basename(args.file_path)
                save_path = folder_name + '_ajusted'

                if os.path.exists(save_path):
                    new_folder_name = save_path + str(datetime.datetime.now())[0:9]
                    save_path = os.path.join(parent_folder, new_folder_name)
            else:
                parent_folder = os.path.dirname(args.file_path)
                file_name = os.path.splitext(os.path.basename(args.file_path))[0]
                output_file = file_name + "_adjusted.nii"
                Save_path = os.path.join(parent_folder, output_file)

        if args.batch_process == "True" and not os.path.exists(save_path):
            os.makedirs(save_path)
        if args.batch_process == "True":
            file_paths = []
            for subfile in os.listdir(args.file_path):
                file_path = os.path.join(args.file_path, subfile)
                file_paths.append(file_path)
            p_list = []
            for nii_path in file_paths:
                pre_path = os.path.splitext(os.path.basename(nii_path))[0]
                Save_path_pre = pre_path + "_adjusted.nii"
                Save_path = os.path.join(save_path, Save_path_pre)
                p = Process(target=self.run_main, args=(args, Save_path, nii_path))
                p_list.append(p)
            for p in p_list:
                p.start()
                p.join()

            if len(file_paths) == 0:
                print(
                    "Sorry, the folder not contain supported files, only the following formats are supported: nii.gz, mha, nii")
        else:
            if ".nii.gz" in args.file_path or ".mha" in args.file_path or ".nii" in args.file_path:

                self.run_main(args, Save_path, args.file_path)
            else:
                print("Sorry, only the following formats are supported: nii.gz, mha, nii")
    def run_main(self, args, Save_path, nii_path):
        self.pre_data(args, nii_path, Save_path)
        self.openfiles()
        self.predict()
        self.adjust()
        self.saveAs()
        time.sleep(1)

    def saveAs(self):
        if self.isOpen:
            if self.adjusted or self.direction == "000":
                savePath = self.Save_path
                save_img = np.zeros((self.data.shape[2], self.data.shape[0], self.data.shape[1]))
                for i in range(self.adjust_img.shape[2]):
                    save_img[i, :, :] = self.data[:, :, i]
                reply = self.keep_header
                if reply:
                    self.DIrection = self.Direction
                else:
                    self.DIrection = self.DIrection
                DIrection = self.DIrection
                img_save = GetImageFromArray(save_img)
                img_save.SetDirection(DIrection)
                img_save.SetOrigin(self.origin)
                img_save.SetSpacing(self.spacing)
                WriteImage(img_save, savePath)
                print(savePath,'has saved.')
if __name__ == '__main__':
    p = Process(target=orderline)
    p.start()
    p.join()
