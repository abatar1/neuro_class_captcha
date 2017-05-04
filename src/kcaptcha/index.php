<?php

error_reporting (E_ALL);

include('kcaptcha.php');

session_start();

$captcha = new KCAPTCHA();

file_put_contents("key", $captcha->getKeyString());
?>