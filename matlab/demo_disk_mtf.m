% Purpose: calculate MTF from the ACR Disk Module phantom scans

clear all; 

file_path = './../data/z_I0_0072000_noiseless_disk.raw';

loc=[225 97; 97 97;  97 225;  225 225]; %HU 340, 150 100 80
hu_num=[1955, 905, 1120, 6];
nloc= size(loc,1);

roisz = 50;
roi = [-roisz/2:roisz/2];
pixelsz = 0.6641; %See code makeCT_ACR_module1.m;
nx=320;



% Open file and read
fid = fopen(file_path, 'r'); % Add 'r' for read-only mode
if fid == -1
   error(['Cannot open file: ' file_path]);
end
img = fread(fid, [nx nx], 'int16');
fclose(fid);

for j=1:nloc
    % j
    %Crop the disk ROI
    imgdisk = double(img(loc(j,1)+roi, loc(j,2)+roi)); %change from unit16 to double

    [mtf, freq, esf, success] = MTF_from_disk_edge(imgdisk);
    freq_vector = freq/pixelsz;

    disp(['HU ' num2str(hu_num(j))]);
    disp(['mtf50% ' num2str(MTF_width(mtf, 0.5, freq_vector))]);
    disp(['mtf20% ' num2str(MTF_width(mtf, 0.2, freq_vector))]);
    disp(['mtf10% ' num2str(MTF_width(mtf, 0.1, freq_vector))]);

end