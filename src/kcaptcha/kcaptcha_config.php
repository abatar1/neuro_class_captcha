<?php
# KCAPTCHA configuration file
$alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"; # do not change without changing font files!

# symbols used to draw CAPTCHA
$allowed_symbols = "23456789abcdegkmnpqsuvxy";

# folder with fonts
$fontsdir = 'fonts';	

# CAPTCHA string length
$length = 1;

# CAPTCHA image size (you do not need to change it, this parameters is optimal)
$width = 160;
$height = 80;

# symbol's vertical fluctuation amplitude
$fluctuation_amplitude = 8;

#noise
$white_noise_density=1/6;
$black_noise_density=1/30;

# increase safety by prevention of spaces between symbols
$no_spaces = true;

# show credits
$show_credits = false; # set to false to remove credits line. Credits adds 12 pixels to image height
$credits = 'www.captcha.ru'; # if empty, HTTP_HOST will be shown

# CAPTCHA image colors (RGB, 0-255)
$foreground_color = array(mt_rand(0,80), mt_rand(0,80), mt_rand(0,80));
$background_color = array(mt_rand(220,255), mt_rand(220,255), mt_rand(220,255));

# JPEG quality of CAPTCHA image (bigger is better quality, but larger file size)
$jpeg_quality = 90;
?>