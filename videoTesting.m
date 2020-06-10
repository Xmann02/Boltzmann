
%test1();
test2();


function test2()
[images, labels] = mnist_parse('train-images.idx3-ubyte','train-labels.idx1-ubyte');

v = VideoWriter('testVideo.avi');
open(v);

for k=1:200
    subplot(1,2,1);
    imshow(images(:,:,2*k));
    subplot(1,2,2);
    imshow(images(:,:,2*k+1));
    
    frame = getframe(gcf);
    writeVideo(v,frame);
end    

close(v);

end



function test1()
close all

axis tight manual

v = VideoWriter('testVideo.avi')
open(v)

[X,Y] = meshgrid(-10:0.5:10, -10:0.5:10);
r = sqrt(X.^2+Y.^2);

for k=0:200
    Z = cos(r/2+k/10).*exp(-r.^2/500);
    surf(X,Y,Z);
    xlim([-10,10]);
    ylim([-10,10]);
    zlim([-1,1]);
    frame = getframe(gcf);
    writeVideo(v,frame);
    
end    
close(v)
end
