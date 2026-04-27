from setuptools import setup, find_packages
setup(
    name="codefix-env",
    version="1.0.0",
    packages=find_packages(),
    install_requires=["fastapi", "uvicorn", "pydantic", "openenv-core"],
)
