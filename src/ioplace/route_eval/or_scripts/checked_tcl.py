"""Fail-fast execution for OpenROAD commands through its Python binding."""


def checked_eval(design, command, label="command"):
    """Exit the OpenROAD process if ``command`` raises a Tcl error.

    Some OpenROAD Python bindings do not translate Tcl failures into Python
    exceptions.  Catching inside Tcl and exiting there makes failure visible
    to the host process regardless of binding behavior.
    """
    if any(char in label for char in "\"\n\r"):
        raise ValueError("invalid Tcl command label")
    script = (
        f"set __ioplace_code [catch {{{command}}} __ioplace_message "
        "__ioplace_options]\n"
        "if {$__ioplace_code != 0} {\n"
        f"  puts stderr \"IOPLACE_TCL_ERROR {label}: $__ioplace_message\"\n"
        "  exit 1\n"
        "}\n"
    )
    return design.evalTclString(script)
