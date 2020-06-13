function ret = ellipseMatrix(y0, x0, a, b, theta, im, c1, c2, pInterp)
% Set the elements of the matrix im which are in the interior of the
% ellipse E to the value 'c1'. The ellipse E has center (y0, x0), the
% major axis = a, the minor axis = b, and theta is the angle made by the
% major axis with the orizontal axe.
% ret = ellipseMatrix(y0, x0, a, b, theta, Image, color)
% ret is a matrix of the same size as Image
%
% The input parameters c2 and pInterp are optional. If they are present 
% the ellipse will have a smoother border (of size pInterp pixels) with 
% variations of % color between c1 and c2.
%
% Function:  modified ellipseMatrixc creation
% Original Author: Nicolae Cindea
[ny, nx] = size(im);        % resolution of the image
imtemp = zeros(ny, nx);     % an empty image 
list = zeros(ny * nx, 2);
toplist = 1;
c = sqrt(a * a - b * b);
x0 = round(x0);
y0 = round(y0);
list(toplist, 1) = y0;
list(toplist, 2) = x0;
im(y0, x0) = c1;
imtemp(y0, x0) = c1;
directions = [0, 1; -1, 0; 0, -1; 1, 0];
while (toplist > 0)
    y = list(toplist, 1);
    x = list(toplist, 2);
    toplist = toplist - 1;
    
    for i = 1:4
        xi = x + directions(i, 1);
        yi = y + directions(i, 2);
        isV = p_isValid(yi, xi, y0, x0, a, c, theta, imtemp, ny, nx, c1, pInterp);
        if isV ~= 0
            toplist = toplist + 1;
            list(toplist, :) = [yi, xi];
            col =  round(isV * c1 + (1-isV)*c2);
            im(list(toplist, 1), list(toplist, 2)) = col;
            imtemp(list(toplist, 1), list(toplist, 2)) = c1;
        end
    end
      
end
ret = im;
%--------------------------------------------------------------------------
function is_val = p_isValid(y, x, y0, x0, a, c, theta, im, ny, nx, c1, pInterp)
d1 = (x - x0 - c * cos(theta))^2 + (y - y0 - c * sin(theta))^2;
d1 = sqrt(d1);
d2 = (x - x0 + c * cos(theta))^2 + (y - y0 + c * sin(theta))^2;
d2 = sqrt(d2);
if (d1 + d2 <= 2*a) && (im(y, x) ~= c1) && (x>0) && (y>0) && ...
        (x <= nx) && (y <= ny)
    is_val = 1;
else
    if (d1 + d2 > 2*a) && (im(y, x) ~= c1) && (x>0) && (y>0) && ...
        (x <= nx) && (y <= ny) && (pInterp ~= 0)
        is_val = abs(min(pInterp, abs(d1 + d2 - 2*a)) - pInterp)/pInterp;
    else
        is_val = 0;
    end
    %disp(is_val)
end