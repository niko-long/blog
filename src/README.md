---
home: true
icon: house
title: 主页
heroImage: logo-dark.png
heroImageDark: /logo-dark.png
bgImage: /bg1.webp
bgImageDark: /bg5.webp
heroFullScreen: true
bgImageStyle:
  background-attachment: fixed
heroText: CodeLong的项目文档
tagline: 在这里可以快速了解我的项目
actions:
  - text: 关于我
    link: ./about/index
    type: primary

highlights:
  - header: ⭐⭐⭐⭐⭐百万级网关系统
    bgImage: /bg6.webp
    bgImageDark: /bg2.webp
    features:
      - title: 项目概述
        icon: fa6-brands:markdown
        details: 分布式微服务网关，支持百万级并发请求处理，提供请求限流,鉴权,熔断等核心能力
      - title: 技术亮点
        icon: code
        details: 自定义SDK + Netty + 网关中心 实现动态路由配置
      - title: 项目地址
        icon: lightbulb
        details: 在GitHub上查看该项目
        link: https://github.com/niko-long/gateway
      - title: 详细信息
        icon: box-archive
        details: 查看详情
        link: ./projects/gateway/README.md

  - header: ⭐⭐⭐⭐畅购通购票系统
    bgImage: /bg3.webp
    bgImageDark: /bg4.webp
    features:
      - title: 系统简介
        icon: fa6-brands:markdown
        details: 高性能票务交易平台，实现秒杀、座位锁定、延迟关单等复杂业务场景
      - title: 架构特点
        icon: network-wired
        details: 提供本地锁+分布式锁和无锁化方案优化购票服务，利用多级缓存架构与延迟队列自动关单提升系统吞吐量。
      - title: 详细信息
        icon: box-archive
        details: 查看详情
        link: ./projects/easypass/README.md

  - header: ⭐⭐⭐山西大学算法队Online Judge系统
    bgImage: /bg6.webp
    bgImageDark: /bg2.webp
    features:
      - title: 系统描述
        icon: fa6-brands:markdown
        details: 支持 ACM/ICPC 模式，提供实时评测、比赛管理、题目管理等功能
      - title: 技术特色
        icon: chart-simple
        details: 使用 Docker/判题机 实现多语言沙箱环境，支持 15+ 种编程语言评测
      - title: 详细信息
        icon: box-archive
        details: 查看详情
        link: ./projects/algorithm/README.md

  - header: ⭐Markdown 渲染
    bgImage: /bg3.webp
    bgImageDark: /bg4.webp
    features:
      - title: 功能说明
        icon: fa6-brands:markdown
        details: 支持 CommonMark 规范扩展，实现流程图、数学公式、交互式代码块等特色功能
      - title: 集成能力
        icon: puzzle-piece
        details: 支持 VuePress、VitePress 等主流文档框架的插件化集成
      - title: 详细信息
        icon: box-archive
        details: 查看详情
        link: ./projects/markdown/README.md

---
