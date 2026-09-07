function rng(seed)
% Compatibility for the rng(seed) calls used by the upstream MATLAB demo.
% Octave 6.4 lacks rng; its state API selects and seeds the Twister generator.
% This does NOT promise MATLAB-identical draws (especially for randn).
% Export actual initial states for comparisons across runtimes.
    rand('state', seed);
    randn('state', seed);
end
