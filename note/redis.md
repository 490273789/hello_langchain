# Redis 学习笔记

## 1. Redis 是什么

Redis 的全称是 **Remote Dictionary Server**，本质上是一个基于内存的高性能键值数据库。

它的典型特点：

- 数据主要存放在内存中，读写速度非常快
- 支持多种数据结构，不只是简单的 `key-value`
- 支持持久化，可以把内存数据落盘
- 支持过期时间、发布订阅、事务、Lua 脚本、Stream
- 常用于缓存、会话存储、排行榜、消息队列、计数器等场景

一句话理解：

> Redis = 高性能内存数据库 + 丰富数据结构 + 常用于缓存和高并发场景

## 2. Redis 为什么快

Redis 性能高，核心原因主要有下面几项：

### 2.1 基于内存

Redis 把数据存储在内存中，内存访问速度远快于磁盘。

### 2.2 单线程处理命令

Redis 处理命令的核心模型长期以来是单线程事件循环。

优点：

- 避免了多线程频繁加锁和上下文切换
- 实现简单，执行路径短
- 对大多数网络型 IO 场景效率很高

注意：

- Redis 不是“只有一个线程”
- 持久化、网络读写等部分在现代版本中可以使用后台线程
- 命令执行主线程通常仍然是核心

### 2.3 IO 多路复用

Redis 使用 IO 多路复用机制，可以同时监听大量客户端连接，在单线程下也能高效处理高并发请求。

### 2.4 数据结构设计高效

Redis 内部针对不同数据类型做了很多优化，比如压缩列表、跳表、哈希表等，使得常见操作复杂度较低。

## 3. Redis 适合做什么

Redis 最常见的使用场景：

- 缓存热点数据
- 分布式锁
- Session 会话共享
- 排行榜
- 点赞、关注、签到、计数器
- 延迟队列或轻量消息队列
- 实时消息流处理

不太适合的场景：

- 超大规模冷数据长期存储
- 强事务、复杂关联查询场景
- 完全替代关系型数据库

结论：

> Redis 更适合做“高性能辅助存储”，不是所有业务数据都应该放进 Redis。

## 4. Redis 的核心数据类型

Redis 的强大之处就在于支持多种数据结构。

### 4.1 String

最基础、最常用的数据类型。

适合：

- 缓存对象 JSON
- 计数器
- Token
- 分布式锁标记

常用命令：

```bash
SET name "ethan"
GET name
INCR page:view
DECR stock:1001
SETEX login:token 300 "abc123"
```

说明：

- `INCR/DECR` 常用于计数
- `SETEX` 用于设置值并附带过期时间
- 一个 key 对应一个 value，value 可以是字符串、数字、JSON 序列化结果等

### 4.2 Hash

Hash 很适合存储对象类型数据。

例如用户信息：

```bash
HSET user:1 name "Tom" age 18 city "Beijing"
HGET user:1 name
HGETALL user:1
HINCRBY user:1 age 1
```

适合：

- 用户资料
- 配置项
- 对象属性集合

优点：

- 比直接存整个 JSON 更适合更新局部字段

### 4.3 List

List 是双向链表结构，适合按插入顺序存储数据。

常用命令：

```bash
LPUSH tasks "task1"
LPUSH tasks "task2"
RPOP tasks
LRANGE tasks 0 -1
```

适合：

- 消息队列的简单实现
- 最新消息列表
- 时间线数据

注意：

- 现代 Redis 中，如果要做更可靠的消息流处理，通常更推荐 `Stream`

### 4.4 Set

Set 是无序且元素唯一的集合。

常用命令：

```bash
SADD tags "redis" "database" "cache"
SMEMBERS tags
SISMEMBER tags "redis"
SREM tags "cache"
```

适合：

- 去重
- 共同好友
- 标签系统
- 用户关注集合

### 4.5 Sorted Set

Sorted Set 在 Set 的基础上，为每个元素增加一个分数 `score`，并按分数排序。

常用命令：

```bash
ZADD ranking 100 user:1 88 user:2 95 user:3
ZRANGE ranking 0 -1 WITHSCORES
ZREVRANGE ranking 0 2 WITHSCORES
ZINCRBY ranking 10 user:2
```

适合：

- 排行榜
- 热度榜
- 延迟任务
- 优先级队列

### 4.6 Stream

Stream 是 Redis 提供的消息流数据结构，适合实现消息队列和消费组。

常用命令：

```bash
XADD orders * user_id 1 amount 99
XRANGE orders - +
XGROUP CREATE orders order_group 0
XREADGROUP GROUP order_group c1 STREAMS orders >
```

适合：

- 消息队列
- 事件流
- 多消费者消费模型

特点：

- 支持消息持久化
- 支持消费组
- 比 `Pub/Sub` 更适合需要确认消费的场景

### 4.7 其他特殊类型

Redis 还支持一些特殊结构：

- `Bitmap`：适合签到、活跃统计
- `HyperLogLog`：适合近似去重统计 UV
- `Geo`：适合附近的人、地理位置检索
- `Bitfield`：位级别操作

## 5. Key 管理与过期机制

Redis 中每个数据都通过 key 访问，因此 key 设计非常重要。

### 5.1 Key 命名建议

推荐使用分层结构：

```text
user:1001:profile
order:20260310:10001
cache:article:888
```

好处：

- 可读性强
- 方便按业务分类
- 便于排查问题

### 5.2 过期时间

Redis 支持为 key 设置 TTL。

常用命令：

```bash
SET code "9527"
EXPIRE code 60
TTL code
PERSIST code
```

典型用途：

- 验证码
- 登录状态
- 缓存数据
- 临时锁

### 5.3 Redis 如何删除过期键

Redis 不是到时间就立刻删除，而是采用组合策略：

- 惰性删除：访问 key 时发现过期再删除
- 定期删除：后台定期随机抽查过期 key 并清理

这样做是为了在 CPU 消耗和内存释放之间做平衡。

## 6. Redis 持久化

Redis 是内存数据库，但为了防止重启丢数据，需要持久化。

Redis 常见持久化方式有两种：

### 6.1 RDB

RDB 是在某个时间点把内存数据快照保存到磁盘。

优点：

- 文件紧凑，恢复快
- 适合备份
- 对运行时性能影响相对较小

缺点：

- 两次快照之间的数据可能丢失

一句话理解：

> RDB 类似“定时拍照”。

### 6.2 AOF

AOF 会把写命令按追加日志的方式保存下来，重启时重新执行这些命令恢复数据。

优点：

- 数据更安全
- 丢失的数据更少

缺点：

- 文件通常比 RDB 大
- 恢复速度通常比 RDB 慢

一句话理解：

> AOF 类似“记操作日志”。

### 6.3 RDB 与 AOF 对比

| 维度 | RDB | AOF |
| --- | --- | --- |
| 原理 | 快照 | 追加写命令 |
| 数据安全性 | 较低 | 较高 |
| 文件体积 | 较小 | 较大 |
| 恢复速度 | 较快 | 较慢 |
| 适合场景 | 备份、快速恢复 | 更高数据可靠性 |

实际生产中常见做法：

- 同时开启 RDB 和 AOF
- 兼顾恢复速度与数据安全

## 7. Redis 内存淘汰策略

当 Redis 内存达到上限后，就需要决定删除哪些数据。

常见策略：

- `noeviction`：不淘汰，写入报错
- `allkeys-lru`：所有 key 中淘汰最近最少使用的
- `volatile-lru`：只在设置了过期时间的 key 中淘汰最近最少使用的
- `allkeys-random`：所有 key 中随机淘汰
- `volatile-random`：在过期 key 中随机淘汰
- `volatile-ttl`：优先淘汰快要过期的 key
- `allkeys-lfu`：所有 key 中淘汰最不经常使用的
- `volatile-lfu`：在过期 key 中淘汰最不经常使用的

缓存场景里常见选择：

- `allkeys-lru`
- `allkeys-lfu`

## 8. Redis 事务、Lua 与原子性

### 8.1 Redis 事务

Redis 事务使用：

- `MULTI`
- `EXEC`
- `DISCARD`
- `WATCH`

示例：

```bash
MULTI
SET balance 100
INCRBY balance 20
EXEC
```

注意：

- Redis 事务不等同于关系型数据库事务
- 它不能像 MySQL 那样完整支持回滚
- Redis 更强调把一组命令按顺序、串行执行

### 8.2 为什么说 Redis 命令是原子的

因为 Redis 在单线程命令执行模型下，一条命令执行过程中不会被其他命令打断。

例如：

- `INCR`
- `HINCRBY`
- `LPUSH`

这些单条命令天然具备原子性。

### 8.3 Lua 脚本

Lua 脚本可以把多个操作打包成一个原子执行单元。

适合：

- 扣库存
- 分布式锁校验与释放
- 复杂原子逻辑

示意：

```lua
if redis.call('GET', KEYS[1]) == ARGV[1] then
	return redis.call('DEL', KEYS[1])
else
	return 0
end
```

这是释放分布式锁时的经典写法，避免误删别人的锁。

## 9. 发布订阅与 Stream

### 9.1 Pub/Sub

Redis 支持发布订阅。

常用命令：

```bash
SUBSCRIBE news
PUBLISH news "hello redis"
```

特点：

- 实现简单
- 实时性高
- 订阅者离线时，消息通常不会补发

适合：

- 即时通知
- 轻量实时消息

### 9.2 Stream 与 Pub/Sub 的区别

`Stream` 更适合业务型消息队列，因为它支持：

- 消息持久化
- 消费组
- 消费确认
- 历史消息回放

结论：

- 简单广播用 `Pub/Sub`
- 可靠消费用 `Stream`

## 10. Redis 高可用

### 10.1 主从复制

Redis 可以配置主节点和从节点。

特点：

- 主节点负责写
- 从节点复制主节点数据
- 从节点可分担读压力

问题：

- 主节点挂了后，需要故障转移

### 10.2 Sentinel

Sentinel 是 Redis 的哨兵机制，用于监控和自动故障转移。

它能做的事：

- 监控主从节点状态
- 发现主节点故障
- 自动选举新主节点
- 通知客户端主节点变更

适合：

- 需要高可用，但数据量还没大到必须分片

### 10.3 Cluster

Redis Cluster 用于分布式分片。

特点：

- 数据分布在多个节点上
- 使用哈希槽机制，总共 `16384` 个槽
- 解决单机内存和吞吐瓶颈

适合：

- 数据量大
- 并发高
- 单机放不下

区别总结：

- `Sentinel` 重点是高可用
- `Cluster` 同时解决高可用和水平扩展问题

## 11. Redis 常见缓存问题

Redis 最常见的实际应用就是缓存，因此要理解几个高频问题。

### 11.1 缓存穿透

请求的数据在数据库和 Redis 中都不存在，导致请求每次都打到数据库。

解决思路：

- 缓存空值
- 布隆过滤器
- 做好参数校验

### 11.2 缓存击穿

某个热点 key 突然过期，大量请求同时打到数据库。

解决思路：

- 热点数据不过期
- 互斥锁
- 提前刷新缓存

### 11.3 缓存雪崩

大量 key 在同一时间失效，导致大量请求直接冲击数据库。

解决思路：

- 过期时间加随机值
- 多级缓存
- 限流降级
- 热点数据预热

## 12. 分布式锁

Redis 常被用来实现分布式锁。

### 12.1 基本写法

```bash
SET lock:order:1 uuid-123 NX EX 30
```

含义：

- `NX`：只有 key 不存在时才设置成功
- `EX 30`：30 秒后自动过期

这条命令可以作为加锁操作。

### 12.2 为什么释放锁不能直接 DEL

如果锁过期后被别人拿到，此时你再执行 `DEL`，可能把别人的锁删掉。

正确做法：

- 给锁设置唯一值
- 删除前先校验 value
- 使用 Lua 脚本保证比较和删除的原子性

### 12.3 Redlock

Redlock 是 Redis 官方提出的一种多节点分布式锁方案。

学习阶段先理解两点即可：

- 单机 Redis 锁实现简单，适合很多普通场景
- 高可靠分布式锁设计比表面看起来复杂得多，需要结合业务容错要求判断是否真的需要

## 13. Redis 常用命令速查

### 13.1 Key 操作

```bash
KEYS *
EXISTS user:1
DEL user:1
EXPIRE user:1 60
TTL user:1
TYPE user:1
```

注意：

- 生产环境中尽量少用 `KEYS *`
- 大数据量下更推荐 `SCAN`

### 13.2 String 操作

```bash
SET k1 v1
GET k1
MSET k2 v2 k3 v3
INCR counter
APPEND k1 "-new"
```

### 13.3 Hash 操作

```bash
HSET user:1 name Tom age 18
HGET user:1 name
HMGET user:1 name age
HGETALL user:1
HDEL user:1 age
```

### 13.4 List 操作

```bash
LPUSH queue a b c
RPUSH queue d
LPOP queue
RPOP queue
LRANGE queue 0 -1
```

### 13.5 Set 操作

```bash
SADD s1 a b c
SMEMBERS s1
SINTER s1 s2
SUNION s1 s2
SDIFF s1 s2
```

### 13.6 Sorted Set 操作

```bash
ZADD rank 99 user1 88 user2
ZSCORE rank user1
ZRANGE rank 0 -1 WITHSCORES
ZREVRANK rank user1
ZREM rank user2
```

## 14. Redis 与 MySQL 的区别

| 维度 | Redis | MySQL |
| --- | --- | --- |
| 存储位置 | 主要在内存 | 主要在磁盘 |
| 查询方式 | key-value 为主 | SQL 关系查询 |
| 性能 | 高并发、低延迟 | 复杂查询能力强 |
| 数据结构 | 丰富 | 以表结构为核心 |
| 事务能力 | 较弱 | 完整事务更强 |
| 适合场景 | 缓存、计数、排行榜 | 核心业务数据 |

常见组合方式：

- MySQL 存核心业务数据
- Redis 存热点缓存、临时状态和高频访问数据

## 15. Redis 使用建议

### 15.1 不要滥用 Redis

Redis 很快，但内存昂贵，不适合存所有数据。

### 15.2 设计合理的 key

建议：

- 统一前缀
- 带业务含义
- 控制 key 长度

### 15.3 注意大 key 问题

大 key 会导致：

- 删除阻塞
- 网络传输变慢
- 内存使用不均衡

例如：

- 一个超大的 Hash
- 一个包含几十万元素的 List 或 Set

### 15.4 注意热 key 问题

某些 key 访问量极高时，会成为性能瓶颈。

常见应对：

- 本地缓存
- 多副本分担
- 热点隔离

### 15.5 使用 SCAN 替代 KEYS

在生产环境下，`KEYS *` 可能阻塞 Redis。

更推荐：

```bash
SCAN 0 MATCH user:* COUNT 100
```

## 16. Python 中使用 Redis

Python 里常用的客户端库是 `redis-py`。

安装：

```bash
pip install redis
```

示例：

```python
import redis

r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

r.set("name", "ethan")
print(r.get("name"))

r.hset("user:1", mapping={"name": "Tom", "age": 18})
print(r.hgetall("user:1"))

r.zadd("ranking", {"user:1": 100, "user:2": 88})
print(r.zrevrange("ranking", 0, -1, withscores=True))
```

## 17. 面试和学习中的高频问题

学习 Redis 时，下面这些问题非常高频：

1. Redis 为什么快？
2. Redis 和 MySQL 的区别是什么？
3. Redis 有哪些数据类型？分别适合什么场景？
4. Redis 持久化机制有哪些？
5. 缓存穿透、击穿、雪崩分别是什么？
6. Redis 如何实现分布式锁？
7. Sentinel 和 Cluster 的区别是什么？
8. 为什么生产环境不建议使用 `KEYS *`？
9. Redis 事务和 MySQL 事务有什么不同？

如果把这些问题理解清楚，Redis 的基础就已经比较扎实了。

## 18. 一页总结

Redis 最重要的结论可以压缩成下面几句：

- Redis 是基于内存的高性能键值数据库
- Redis 支持多种数据结构，不只是字符串
- Redis 最常见用途是缓存，也常用于计数器、排行榜、分布式锁和消息系统
- Redis 通过 RDB 和 AOF 实现持久化
- Redis 通过主从复制、Sentinel、Cluster 支持高可用和扩展
- 实际使用中要重点关注缓存问题、过期策略、大 key、热 key 和内存淘汰策略

如果你按下面的顺序学习，效率会比较高：

1. 先掌握 `String/Hash/List/Set/ZSet`
2. 再理解过期机制、持久化、淘汰策略
3. 然后学习缓存问题和分布式锁
4. 最后学习 Sentinel、Cluster、Stream 这些偏工程化内容
