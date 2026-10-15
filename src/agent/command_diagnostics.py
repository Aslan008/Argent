"""
Diagnostics for failed shell commands.

When a command fails, a weak model often misreads the error and invents a wrong
cause (e.g. blaming the C# project when the real problem was a PowerShell cmdlet
run in cmd). diagnose_command_error() recognizes common failure signatures and
returns a concrete, actionable hint that is appended to the command result, so
the model is pointed at the real cause instead of guessing.
"""

import re

# PowerShell cmdlets that, when "not recognized", mean the command ran in cmd.
_PS_CMDLET = re.compile(
    r"\b(?:Select|Where|ForEach|Sort|Measure|Format|Out|Get|Set|New|Remove|Add|"
    r"Test|Invoke|Write|Read|ConvertTo|ConvertFrom)-\w+",
    re.IGNORECASE,
)

_NOT_RECOGNIZED = (
    "is not recognized as an internal or external",
    "не является внутренней или внешней",
    "не является внутренней либо внешней",
    "the term",  # PowerShell: "The term 'x' is not recognized..."
)
_ACCESS_DENIED = ("access is denied", "отказано в доступе", "permission denied")
_PATH_NOT_FOUND = (
    "no such file or directory", "cannot find path", "could not find",
    "не удается найти", "не удалось найти", "не найден путь", "system cannot find",
)


def _contains(text_low: str, needles) -> bool:
    return any(n in text_low for n in needles)


def diagnose_command_error(command: str, output: str, exit_code: int) -> str | None:
    """Return a concrete hint for a recognized command failure, or None.

    `output` is the raw combined stdout/stderr; `command` is the command string.
    """
    if exit_code == 0:
        return None
    low = output.lower()

    # 1. Windows NTSTATUS crash codes & POSIX signals
    nt_crashes = {
        -1073741819: (
            "CRASH (STATUS_ACCESS_VIOLATION 0xC0000005): Segmentation Fault / Memory Access Violation. "
            "The program attempted to read or write an invalid memory address (e.g. null pointer dereference, "
            "use-after-free, vector out-of-bounds, or uninitialized pointer). "
            "Inspect pointer usages or recompile with AddressSanitizer (-fsanitize=address / /fsanitize=address)."
        ),
        3221225477: (
            "CRASH (STATUS_ACCESS_VIOLATION 0xC0000005): Segmentation Fault / Memory Access Violation. "
            "The program attempted to read or write an invalid memory address. "
            "Inspect pointer usages or recompile with AddressSanitizer."
        ),
        -1073741571: (
            "CRASH (STATUS_STACK_OVERFLOW 0xC00000FD): Stack Overflow. "
            "Uncontrolled/infinite recursion or excessive local stack allocations (e.g. huge local arrays). "
            "Allocate large buffers on the heap (std::vector, unique_ptr) or fix recursive base conditions."
        ),
        3221225725: (
            "CRASH (STATUS_STACK_OVERFLOW 0xC00000FD): Stack Overflow. Infinite recursion or huge stack allocation."
        ),
        -1073740791: (
            "CRASH (STATUS_STACK_BUFFER_OVERRUN 0xC0000409): Stack corruption / buffer overrun detected (/GS). "
            "A buffer write exceeded its boundary and overwrote security cookies or function return address."
        ),
        3221226505: (
            "CRASH (STATUS_STACK_BUFFER_OVERRUN 0xC0000409): Stack corruption / buffer overrun detected."
        ),
        -1073741795: (
            "CRASH (STATUS_ILLEGAL_INSTRUCTION 0xC000001D): Illegal CPU instruction executed. "
            "Code compiled for newer CPU instruction sets (e.g. AVX2, AVX-512) executed on incompatible CPU, "
            "or function pointer jumped into data/garbage."
        ),
        139: (
            "CRASH (SIGSEGV 139): Segmentation Fault (Linux/POSIX). Invalid memory access or null pointer dereference. "
            "Run with AddressSanitizer (-fsanitize=address) or inspect core dump."
        ),
        134: (
            "CRASH (SIGABRT 134): Program called abort() or assertion failed (e.g. assert(), unhandled std::terminate). "
            "Check assertion messages printed before termination."
        ),
    }
    if exit_code in nt_crashes:
        return nt_crashes[exit_code]

    # Check for text-embedded NTSTATUS crash signatures (when exit code was wrapped or masked)
    if "0xc0000005" in low or "status_access_violation" in low:
        return nt_crashes[-1073741819]
    if "0xc00000fd" in low or "status_stack_overflow" in low:
        return nt_crashes[-1073741571]
    if "0xc0000409" in low or "status_stack_buffer_overrun" in low:
        return nt_crashes[-1073740791]

    # 2. C / C++ Compiler include errors
    if "cannot open include file" in low or ("no such file or directory" in low and ("fatal error" in low or "#include" in low or "error:" in low)):
        return (
            "C/C++ Include Not Found: The compiler cannot find a header file. "
            "Check the include path and add the directory to target_include_directories in CMakeLists.txt."
        )

    # 3. CMake errors
    if "cmake error" in low:
        return (
            "CMake Configuration Error: Read the CMake error above. Check package names (find_package), "
            "minimum required version, or target dependencies in CMakeLists.txt."
        )

    # 4. C / C++ Linker errors (MSVC, GCC, Clang, ld)
    if "lnk2019" in low or "lnk1120" in low or "undefined reference to" in low or "cannot find -l" in low or "symbol(s) not found" in low:
        return (
            "C/C++ Linker Error (Unresolved External Symbol / Missing Library): A function, variable, or library "
            "was declared or linked but could not be found by the linker. Ensure implementation .cpp files are "
            "compiled, or add the missing library to target_link_libraries in CMakeLists.txt."
        )
    if "lnk2005" in low or "multiple definition of" in low:
        return (
            "C/C++ Linker Error (Duplicate Symbol): A symbol is defined in multiple translation units. "
            "Move function bodies from headers into .cpp files, or declare them 'inline'."
        )

    # PowerShell cmdlet executed in cmd.exe (the classic Select-Object case).
    if _contains(low, _NOT_RECOGNIZED):
        if _PS_CMDLET.search(command):
            m = _PS_CMDLET.search(command)
            return (
                f"The command failed because '{m.group(0)}' is a PowerShell cmdlet but the "
                f"command was run in cmd.exe. Either drop the PowerShell pipeline, or rewrite "
                f"the command in pure PowerShell. The C# code / project is NOT the problem here."
            )
        return (
            "A program or command was not found (see the named token in the error). "
            "Check it is installed and on PATH, or that you spelled it correctly. "
            "This is a COMMAND problem, not a code problem."
        )

    # Needs elevation.
    if _contains(low, _ACCESS_DENIED):
        return (
            "Access was denied — this action needs elevated privileges. Use "
            "`run_admin_command` for it, or choose a path/operation that doesn't require admin."
        )

    # Path/file not found.
    if _contains(low, _PATH_NOT_FOUND):
        return (
            "A file or directory in the command was not found. Verify the path exists "
            "(use list_directory / search_files) before re-running. Watch for quoting of "
            "paths with spaces."
        )

    # C# / .NET compiler errors — the real cause is in the source, with a code.
    if re.search(r"error\s+cs\d+", low):
        codes = ", ".join(sorted(set(c.upper() for c in re.findall(r"cs\d+", low)))[:5])
        return (
            f"The C# compiler reported errors ({codes}). Read the CSxxxx message and the "
            f"file:line it points to, fix that source, then rebuild. The build command itself is fine."
        )

    # MSBuild / dotnet generic.
    if "msb" in low and re.search(r"msb\d+", low):
        return "MSBuild reported an error (MSBxxxx) — read the referenced project/target and fix it."

    # Python traceback.
    if "traceback (most recent call last)" in low:
        return (
            "Python raised an exception. Read the LAST line of the traceback (the exception type "
            "and message) and the file:line just above it — fix that, not the command."
        )

    # npm / node.
    if "npm err!" in low or "enoent" in low:
        return (
            "npm/node error. Check you are in the right directory (package.json present) and that "
            "dependencies are installed (npm install). ENOENT means a file/path is missing."
        )

    # Generic unknown non-zero exit: still nudge to read the output, not the code.
    return None
