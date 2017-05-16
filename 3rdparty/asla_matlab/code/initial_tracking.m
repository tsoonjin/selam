% simple tracking and collecting tracking results as templates
exemplars_stack = [];
%% read first frame
begin = 1;
if exist([dataPath int2str(begin) '.jpg'],'file')
    imgName = sprintf('%s%d.jpg',dataPath,begin);
elseif exist([dataPath int2str(begin) '.bmp'],'file')
    imgName = sprintf('%s%d.bmp',dataPath,begin);
else 
    imgName = sprintf('%s%05d.jpg',dataPath,begin); 
end
frame = imread(imgName);
if size(frame,3)==3
    grayframe = rgb2gray(frame);
else
    grayframe = frame;
    frame = double(frame)/255; 
end
frame_img = double(grayframe)/255; 
result = [result; param0']; % each estimation is a row vector
exemplar = warpimg(frame_img, param0, psize);  
exemplar = exemplar.*(exemplar>0); 
exemplars_stack = [exemplars_stack, exemplar(:)]; % collect exemplars£» notice that these are not normalized using L2 norm
% draw result
drawopt = drawtrackresult([], begin, frame, psize, result(end,:)'); 
imwrite(frame2im(getframe(gcf)),sprintf('result/%s/Result/%04d.jpg',title,begin));
% imwrite(frame2im(getframe(gcf)),sprintf('result/%s/Result/%04d.fig',title,begin));
imwrite(frame2im(getframe(gcf)),sprintf('result/%s/Result/%04d.png',title,begin));

%% simple tracking
for f = begin+1 : begin+EXEMPLAR_NUM-1
    if exist([dataPath int2str(begin) '.jpg'],'file')
        imgName = sprintf('%s%d.jpg',dataPath,f);
    elseif exist([dataPath int2str(begin) '.bmp'],'file')
        imgName = sprintf('%s%d.bmp',dataPath,f);
    else 
        imgName = sprintf('%s%05d.jpg',dataPath,f); 
    end
    frame = imread(imgName);
    if size(frame,3)==3
        grayframe = rgb2gray(frame);
    else
        grayframe = frame;
        frame = double(frame)/255; 
    end  
    frame_img = double(grayframe)/255; 
    
    particles_geo = sampling(result(end,:), opt.numsample, opt.affsig); 
    candidates = warpimg(frame_img, affparam2mat(particles_geo), psize);
    candi_data = reshape(candidates, psize(1)*psize(2), opt.numsample); 
    candi_data = candi_data.*(candi_data>0);  
    
    % use knn function of the vlfeat open source library
    candidate_kdTree = vl_kdtreebuild(candi_data);   
    [idx, distances] = vl_kdtreequery( candidate_kdTree, candi_data, exemplars_stack(:,end), 'NumNeighbors', 1) ;        
    
    result = [result; affparam2mat(particles_geo(:,idx))']; 
    exemplars_stack = [exemplars_stack, candi_data(:,idx)]; 
    
    % draw result
    drawopt = drawtrackresult(drawopt, f, frame, psize, result(end,:)'); % 
    imwrite(frame2im(getframe(gcf)),sprintf('result/%s/Result/%04d.png',title,f));
end