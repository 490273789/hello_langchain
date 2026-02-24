from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

app = FastAPI()

# 路由 - URL地址和处理函数之间的映射关系


@app.get("/")
async def root():
    return {"message": "Hello FastAPI"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


# 路径参数： url路径的一部分 /book/{id}  指向唯一的、特定的资源
# 查询参数：url 问号之后 /book?age=18&name=wsn  对资源集合进行分过滤、排序、分页等操作
# 请求体：在消息体中，作用是创建、更新或携带大量数据 - post/put


@app.get("/book/{id}")
def book_info(id: int = Path(..., gt=0, lt=101, description="书籍id，取值范围1到100")):
    return {"id": id, "message": f"This is no{id} book."}


# 需求“查询新闻 -> skip: 跳过的记录数， limit： 返回的记录数


@app.get("/news-list")
def get_news_list(
    skip: int = Query(default=10, description="跳过数量"),
    limit: int = Query(..., gt=1, lt=100, description="每页限制数"),
):
    return {"skip": skip, "limit": limit}


class User(BaseModel):
    user_name: str
    password: str = Field(..., max_length=20, min_length=6, description="用户密码")


class ResponseRegister(BaseModel):
    user_name: str = Field(description="用户名")


# 自定义响应数据格式
# 请求体
@app.post("/user/register", response_model=ResponseRegister)
def register(user: User):
    if user.user_name == " ":
        raise HTTPException(status_code=501, detail="用户名不能为空格")
    return {"userName": user.user_name}


# 返回类型
# JSONResponse
# HTMLResponse
# FileResponse


# 设置响应类型的方式
# 方式1:通过装饰器定义
@app.get("/html", response_class=HTMLResponse)
def get_html():
    return "<h1>html response</h1>"


# 方式2：返回对应的类型
@app.post("/download")
def download():
    file_path = "./1.md"
    return FileResponse(file_path)
