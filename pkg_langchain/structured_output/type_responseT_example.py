"""
type[ResponseT] 类型注解示例

`type[ResponseT]` 表示：一个类型本身（不是实例），该类型是 ResponseT 或其子类
"""

from typing import TypeVar

from pydantic import BaseModel

# ResponseT 是一个类型变量，可以代表任何类型
ResponseT = TypeVar("ResponseT")


# ============= 示例 1: Pydantic 模型作为 ResponseT =============
class UserInfo(BaseModel):
    """用户信息模型"""

    name: str
    age: int
    email: str


class ProductInfo(BaseModel):
    """产品信息模型"""

    product_id: str
    name: str
    price: float


# ============= 示例 2: 使用 type[ResponseT] 的函数 =============


def process_response_format(response_format: type[ResponseT]) -> str:
    """
    接收一个类型（不是实例）作为参数

    Args:
        response_format: 一个类本身，比如 UserInfo 类（不是 UserInfo 的实例）
    """
    return f"处理的响应格式类型是: {response_format.__name__}"


# ============= 示例 3: 实际使用 =============

if __name__ == "__main__":
    # ✅ 正确用法：传递类本身
    result1 = process_response_format(UserInfo)
    print(result1)  # 输出: 处理的响应格式类型是: UserInfo

    result2 = process_response_format(ProductInfo)
    print(result2)  # 输出: 处理的响应格式类型是: ProductInfo

    print("\n" + "=" * 60 + "\n")

    # ❌ 错误用法：传递实例而不是类
    # user_instance = UserInfo(name="张三", age=25, email="zhangsan@example.com")
    # process_response_format(user_instance)  # 类型检查会报错

    # ============= 示例 4: 在 create_agent 中的实际应用 =============
    """
    在 langchain 的 create_agent 函数中：
    
    create_agent(
        model="gpt-4",
        response_format=UserInfo,  # ← 传递类本身，不是实例
        # response_format=UserInfo(),  # ❌ 错误：不要传递实例
    )
    
    这样 agent 就知道：
    1. 期望的响应结构是 UserInfo
    2. 可以根据这个类的定义来验证和解析响应
    3. 可以使用 UserInfo 的字段信息（name, age, email）来构造提示词
    """

    # ============= 示例 5: type[X] vs X 的区别 =============

    def create_instance(cls: type[UserInfo]) -> UserInfo:
        """
        参数类型是 type[UserInfo]：接收 UserInfo 类本身
        返回类型是 UserInfo：返回 UserInfo 的实例
        """
        return cls(name="默认用户", age=0, email="default@example.com")

    # 调用时传递类
    user = create_instance(UserInfo)
    print(f"\n创建的用户实例: {user}")
    print(f"实例类型: {type(user)}")
    print(f"是否是 UserInfo 的实例: {isinstance(user, UserInfo)}")

    # ============= 示例 6: 泛型版本 =============

    def create_generic_instance(cls: type[ResponseT], **kwargs) -> ResponseT:
        """
        泛型函数：可以创建任何类型的实例

        Args:
            cls: 任何类型的类（type[ResponseT]）
            **kwargs: 构造参数

        Returns:
            ResponseT: 该类型的实例
        """
        return cls(**kwargs)

    # 可以用于不同的类型
    user2 = create_generic_instance(
        UserInfo, name="李四", age=30, email="lisi@example.com"
    )
    print(f"\n创建的用户: {user2}")

    product = create_generic_instance(
        ProductInfo, product_id="P001", name="笔记本电脑", price=5999.99
    )
    print(f"创建的产品: {product}")


# ============= 总结 =============
"""
type[ResponseT] 的关键点：

1. type[X] 表示"类 X 本身"，不是"类 X 的实例"
   - UserInfo 是 type[UserInfo]
   - UserInfo() 是 UserInfo（实例）

2. ResponseT 是类型变量，可以是任何类型
   - type[ResponseT] 意思是：任何类型的类本身

3. 在 create_agent 中的应用：
   response_format: ResponseFormat[ResponseT] | type[ResponseT] | None
   
   意思是 response_format 参数可以接受：
   - ResponseFormat[ResponseT]: 一个包装了响应格式的对象
   - type[ResponseT]: 直接传递一个类（如 UserInfo）
   - None: 不指定响应格式

4. 实际例子：
   create_agent(
       model="gpt-4",
       response_format=UserInfo  # ← 传递类本身
   )
   
   agent 内部可以：
   - 检查 UserInfo 的字段
   - 创建 UserInfo 实例: UserInfo(**parsed_data)
   - 验证响应是否符合 UserInfo 的结构
"""
