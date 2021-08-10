class CustomUserModule(object):
    def __init__(self, *_, **kwargs):
        self.__kwargs__ = kwargs

    def module_arg(
            self, arg_name, instance_check=None, false_val=None, kwargs=None,
            force_user=False, convert_type=None):
        kwargs = self.__kwargs__ if kwargs is None else kwargs
        if convert_type is None:
            def converter(x): return x
        else:
            converter = convert_type
        if force_user is True and arg_name not in kwargs.keys():
            printt("{} is required but not specified by the user".format(
                arg_name), error=True, stop=True)
        return false_val if not(
            arg_name in kwargs.keys() and (
                instance_check is None or (
                    isinstance(
                        kwargs[arg_name], instance_check
                    ) if convert_type is None else True)
            )) else converter(kwargs[arg_name])
