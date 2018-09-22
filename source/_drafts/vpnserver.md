title: Ubuntu 搭建 VPN 服务器 
date: 2015-12-18 16:49:56
categories: 笔记
tags: 
  - Linux
  - VPN
---

### 安装pptpd

``` bash
$ sudo apt-get install pptpd
```
<!-- more -->

### 修改/etc/ppp/chap-secrets文件
    四列分别是用户名、服务类型（与pptpd-options里name保持一致）、密码、IP限制（不做限制写 * 即可）。
    例如: user pptpd passwd *
    
### 修改/etc/ppp/pptpd-options文件
    取消注释 ms-dns ，
    ms-dns 8.8.8.8
    
### 修改/etc/pptpd.conf文件
    设置VPN连通后服务器及客户端的client的ip。
    localip 192.168.10.1
    remoteip 192.168.10.100-120

### 修改/etc/sysctl.conf
    取消注释 net.ipv4.ip_forward=1
    执行sudo /sbin/sysctl -p重新加载配置
    然后使新配置生效：
    /etc/init.d/procps restart
    
### 设置iptables    
    sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    
### 修改/etc/rc.local文件
    在exit 0这行上面加上
    iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE
    
### 最后重启pptpd服务
    sudo /etc/init.d/pptpd restart