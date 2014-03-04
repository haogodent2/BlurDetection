function [edges]=edge_width_measure(I, I_edge)
% assume I is a grayscale image with properly reduced footprint
% figure
% imshow(I)
% %mask
sobel_x = [1 0 -1;2 0 -2;1 0 -1]; % sobel operator for x derivative
I_xdot = conv2(double(I),double(sobel_x),'valid');
[mask_width,mask_height] = size(I_xdot);

I_edge_valid = I_edge(2:end-1,2:end-1);
% calculate all edge widths in x dimension
%     x, y , width

[row,col] = find(I_edge_valid);

edges = zeros(length(row),3);
edges(:,1) = row;
edges(:,2) = col;
padding = 15;

%trim edges inside outside of padding area
for i = 1:length(edges)
    if (edges(i,1) <= padding) || (edges(i,1) >= (mask_width-padding)) ...
            || (edges(i,2) <= padding) || (edges(i,2) >= (mask_width-padding))
        edges(i,3) = NaN;
    end
end
%discard NaN values
   edges(find(isnan(edges(:,3))==1),:)=[];
   size(edges);
   
% scan width of edge for each edge found in edge detector
for edge_index = 1:length(edges(:,1))
    current_row = edges(edge_index,1); % location of current edge in question
    current_col = edges(edge_index,2); % location of current edge in question
    I_edge_valid(current_row,current_col);
    
    I_xdot(current_row,current_col);
    edge_xdot_sign = sign(I_xdot(current_row,current_col));   % sign of derivative of current location
    edge_pixel = 1;
    left_found = 0 ;
    left_pixel = 1;
    left_index = current_col-left_pixel;
    
    
    while ~left_found
        I_xdot(current_row,left_index);
        if sign(I_xdot(current_row,left_index)) ~= edge_xdot_sign
            left_found = 1;
        else
            left_pixel = left_pixel+1;
            left_index = current_col-left_pixel;
            if left_index <= 2
                break
            end
        end
    end
    right_found = 0;
    right_pixel = 1;
    right_index = current_col+right_pixel;
    
    while ~right_found
        I_xdot(current_row,right_index);
        if sign(I_xdot(current_row,right_index)) ~= edge_xdot_sign
            right_found = 1;
        else
            right_pixel = right_pixel+1;
            right_index = current_col+right_pixel;
            if right_index >= mask_width-3
                break
            end
        end
    end
    edges(edge_index,3) = left_pixel+edge_pixel+right_pixel;

end










