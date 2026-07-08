---
author: admin
comments: true
date: 2010-03-26 09:48:35+00:00
layout: post
slug: what-to-do-when-ubuntu-device-mapper-seems-to-be-invincible
title: 'What to do when Ubuntu Device-mapper seems to be invincible! '
categories:
- Instructional
tags:
- 64-bit
- hard drive
- hardware
- linux
- lucid lynx
- Ubuntu
- workstation
---

I've been trying a dozen different configurations of my 2x500GB SATA drives over the past few days involving switching between ACHI/IDE/RAID in my bios (This was after trying different things to solve [my problems with Ubuntu Lucid Lynx]({{ BASE_PATH }}/2010/03/my-experience-with-ubuntu-10-04-lucid-lynx/)) ; After each attempt I've reset the bios option, booted into a live CD, deleting partitions and rewriting partition tables left on the drives.

Now, however, I've been sitting with a /dev/mapper/nvidia_XXXXXXX1 that seems to be impossible to kill!

It's the only 'partition' that I see in the Ubuntu install (but I can see the others in parted) but it is only the size of one of the drives, and I know I did not set any RAID levels other than RAID0.

Thanks to [wazoox](http://perlmonks.org/?node_id=292373) for eliminating a possibility involving LVM issues with lvremove and vgremove, but I found what works for me.

After a bit of experimenting, I tried

> 

>     
>     <code>$dmraid -r
>     </code>
> 
> 

so see what raid sets were set up, then did

> 

>     
>     <code>$dmraid -x
>     </code>
> 
> 

but was presented with

> ERROR: Raid set deletion is not supported in "nvidia" format

Googled this and found [this forum post](http://ubuntuforums.org/showthread.php?p=8417410) that told me to do this;

> 

>     
>     <code>$dmraid -rE
>     </code>
> 
> 

And that went through, rebooted, hoped, waited (well, while i was waiting, set the bios back to AHCI), and repartitioned, and all of (this particular issue) was well again. Hope this helps someone else down the line!

(This is a duplicate of my [ServerFault query](http://serverfault.com/questions/125976/ubuntu-device-mapper-seems-to-be-invincible) on this that I answered myself)
