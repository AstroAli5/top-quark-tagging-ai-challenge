function [image,radial] = buildWinnerJetFeatures(fourVectors,cfg)
%BUILDWINNERJETFEATURES Independently implemented aligned images + radial stats.
% Concepts: Adit Shah's 2025 MathWorks winner; see WINNER_COMPARISON.md.
% Channels: count, sum pT, sum |pz|, sum E, var E, var pT, skew E,
% skew pT, kurt E, kurt pT, mean E, sum(radius * |momentum|).
% Moments are population moments of occupied particles only. Constant and
% single-particle samples have zero standardized moments by convention.
    validateattributes(fourVectors,{'numeric'},{'2d','ncols',4,'finite'});
    validateattributes(cfg.winnerImageSize,{'numeric'},{'scalar','integer','>=',3});
    assert(mod(cfg.winnerImageSize,2) == 1,'Winner image size must be odd.');
    validateattributes(cfg.winnerExtent,{'numeric'},{'scalar','positive','finite'});
    validateattributes(cfg.winnerMaxParticles,{'numeric'},{'scalar','integer','positive'});
    fv = double(fourVectors);
    pt = hypot(fv(:,2),fv(:,3));
    keep = fv(:,1) > 0 & pt > 0;
    fv = fv(keep,:); pt = pt(keep);
    n = cfg.winnerImageSize;
    image = zeros(n,n,12,'single');
    radial = zeros(1,4,'single');
    if isempty(fv), return; end
    [pt,order] = sort(pt,'descend');
    order = order(1:min(numel(order),cfg.winnerMaxParticles));
    fv = fv(order,:); pt = pt(1:numel(order));
    eta = asinh(fv(:,4)./pt);
    phi = atan2(fv(:,3),fv(:,2));
    x = eta-eta(1);
    y = atan2(sin(phi-phi(1)),cos(phi-phi(1)));
    % Rotate continuously so the second-hardest constituent points right.
    if numel(pt) > 1
        angle = atan2(y(2),x(2));
        rotatedX = cos(angle)*x + sin(angle)*y;
        y = -sin(angle)*x + cos(angle)*y;
        x = rotatedX;
    end
    if sum(fv(y < 0,1)) > sum(fv(y > 0,1)), y = -y; end
    edges = linspace(-cfg.winnerExtent,cfg.winnerExtent,n+1);
    col = discretize(x,edges); row = discretize(y,edges);
    inside = ~isnan(col) & ~isnan(row);
    fv = fv(inside,:); pt = pt(inside);
    radius = hypot(x(inside),y(inside));
    pixels = sub2ind([n n],row(inside),col(inside));
    for pixel = reshape(unique(pixels),1,[])
        selected = pixels == pixel;
        e = fv(selected,1); p = pt(selected);
        [ve,se,ke] = moments(e); [vp,sp,kp] = moments(p);
        momentum = sqrt(sum(fv(selected,2:4).^2,2));
        values = [sum(selected),sum(p),sum(abs(fv(selected,4))),sum(e), ...
            ve,vp,se,sp,ke,kp,mean(e),sum(radius(selected).*momentum)];
        for channel = 1:12
            image(pixel+(channel-1)*n*n) = single(values(channel));
        end
    end
    center = (n+1)/2;
    [xx,yy] = meshgrid(1:n,1:n);
    ring = max(abs(xx-center),abs(yy-center))+1;
    eImage = double(image(:,:,4)); pImage = double(image(:,:,2));
    eProfile = accumarray(ring(:),eImage(:),[center 1]);
    pProfile = accumarray(ring(:),pImage(:),[center 1]);
    [~,se,ke] = moments(eProfile); [~,sp,kp] = moments(pProfile);
    radial = single([se ke sp kp]);
end

function [variance,skewness,kurtosis] = moments(values)
    centered = values-mean(values);
    variance = mean(centered.^2);
    skewness = 0; kurtosis = 0;
    if numel(values) > 1 && variance > eps(max(1,mean(values.^2)))
        skewness = mean(centered.^3)/variance^1.5;
        kurtosis = mean(centered.^4)/variance^2; % Pearson, not excess kurtosis
    end
end
