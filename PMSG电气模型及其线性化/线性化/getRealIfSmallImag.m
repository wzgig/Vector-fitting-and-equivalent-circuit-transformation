function y = getRealIfSmallImag(x, tolAbs, tolRel, action)
% GETREALIFSMALLIMAG  Drop tiny imaginary parts; keep others.
% y = getRealIfSmallImag(x)           % 默认阈值：tolAbs=1e-9, tolRel=1e-12
% y = getRealIfSmallImag(x, tolAbs, tolRel)
% y = getRealIfSmallImag(x, tolAbs, tolRel, action)
% action: 'warn'（默认），'error'，'keep'
%
% 逻辑：若 |Im(x)| <= max(tolAbs, tolRel*max(1,|x|)) 则 y = real(x)，否则按 action 处理。

    if nargin < 2 || isempty(tolAbs), tolAbs = 1e-9; end
    if nargin < 3 || isempty(tolRel), tolRel = 1e-12; end
    if nargin < 4 || isempty(action), action = 'warn'; end

    % 用 double 做阈值判断，不改变原始精度
    abs_im = abs(imag(double(x)));
    mag    = max(1, abs(double(x)));
    thresh = max(tolAbs, tolRel .* mag);

    smallImag = abs_im <= thresh;

    y = x;
    % 对"虚部很小"的元素，直接取实部
    y(smallImag) = real(x(smallImag));

    % 对"虚部不小"的元素，按 action 处理
    if any(~smallImag, 'all')
        switch lower(action)
            case 'warn'
                warning('getRealIfSmallImag:NonnegligibleImag', ...
                        'Non-negligible imaginary part detected (max|Im|=%g). Keeping complex value.', ...
                        max(abs_im(~smallImag)));
            case 'error'
                error('getRealIfSmallImag:NonnegligibleImag', ...
                      'Non-negligible imaginary part (max|Im|=%g).', ...
                      max(abs_im(~smallImag)));
            case 'keep'
                % 什么也不做，保留复数
            otherwise
                warning('Unknown action "%s". Using "warn".', action);
        end
    end
end
