from typing import TypeVar, Optional, cast, Callable


class TypeUtils:
    T = TypeVar('T')

    @staticmethod
    def unbox(optional: Optional[T], throw_exception=True) -> T:
        if optional is None and throw_exception:
            raise ValueError('Variable of type' + type(optional).__name__ + ' is None')
        return cast(TypeUtils.T, optional)
        pass

    @staticmethod
    def if_present(optional: Optional[T], callback: Callable[[T], None]) -> None:
        if optional is not None:
            callback(TypeUtils.unbox(optional))
        pass

    pass
