"""Some helper functions to give interactive talks"""
import os
import sys
from hashlib import sha256
from subprocess import run, PIPE
from time import process_time

import numpy as np

from IPython.display import HTML
from IPython.core.magic import register_cell_magic, register_line_cell_magic
from IPython.display import display


@register_line_cell_magic
def question( line, cell ):
    """Show a question in html and also /easypoll command for slack Easy Poll app.
    Example usage:
    _______________
    %%question Q1

    What is the answer to the ultimate question?
    42|17|Don't know
    __________________

    Notice that cell should contain exactly 4 lines in cluding %%question line and a blank line
    after that
    """
    lines = cell.split( "\n" )
    assert len( lines ) == 4, f"need exactly two lines got {repr( cell )}"
    question = lines[1]
    answers = lines[2]
    show_question( line, question, answers )


def show_question( number: str, question: str,  answers: str ):
    "Show a question in html and also easypoll command."
    answers_arr = answers.split('|')
    assert len(answers_arr) <= 7

    easy_poll_command = [ f'<p> /easypoll "Question {number}.  {question}" '
                          + " ".join(f'"{ans}"' for i, ans in enumerate(answers_arr ))
                          + "</p>" ]

    question_html = [f"<p> <b>Question {number}.</b> {question}</p>"]
    answers_html = [f"<li>{ans}</li>" for ans in answers_arr ]
    whole_html = easy_poll_command + question_html + ['<ol>'] + answers_html + ['</ol']
    display( HTML( "\n".join(whole_html) ) )


def run_c_program( code: str ):
    """compile and run a c program"""
    code_hash = sha256( code.encode( "utf-8" ) ).hexdigest()[:16]
    source_fpath = f"./tmp_{code_hash}.c"
    with open( source_fpath, "wt" ) as f_out:
        f_out.write( code )

    executable_fpath = source_fpath.replace( '.c', '' )

    compilation_result = run( ['gcc', source_fpath, '-lm', '-o', executable_fpath],
                              stdout=PIPE, stderr=PIPE )
    if compilation_result.returncode != 0:
        print( "ERROR: compilation failed:\n"
               + compilation_result.stderr.decode( "utf-8" ), file=sys.stderr )

    exec_result = run( [executable_fpath], stdout=PIPE, stderr=PIPE )

    print( exec_result.stdout.decode( "utf-8" ) )
    os.unlink( source_fpath )
    os.unlink( executable_fpath )


@register_cell_magic
def run_c_code( line: str, cell: str ):
    "my cell magic"
    run_c_program( cell )


@register_line_cell_magic
def run_ruby_program( line: str, code: str ):
    """compile and run a c program"""
    code_hash = sha256( code.encode( "utf-8" ) ).hexdigest()[:16]
    source_fpath = f"./tmp_{code_hash}.rb"
    with open( source_fpath, "wt" ) as f_out:
        f_out.write( code )

    exec_result = run( ["ruby", "-v", source_fpath], stdout=PIPE, stderr=PIPE )

    print( exec_result.stdout.decode( "utf-8" ) )

    if exec_result.returncode != 0:
        print( exec_result.stderr.decode( "utf-8" ) )

    if line != 'keep':
        os.unlink( source_fpath )

@register_cell_magic
def run_crystal_program( line: str, code: str ):
    """compile and run a c program"""
    code_hash = sha256( code.encode( "utf-8" ) ).hexdigest()[:16]
    source_fpath = f"./tmp_{code_hash}.cr"
    with open( source_fpath, "wt" ) as f_out:
        f_out.write( code )

    exec_result = run( ['crystal', 'run', source_fpath],
                       stdout=PIPE, stderr=PIPE )

    if exec_result.returncode != 0:
        print( "ERROR:\n" + exec_result.stderr.decode( "utf-8" ),
               file=sys.stderr )

    print( exec_result.stdout.decode( "utf-8" ) )
    os.unlink( source_fpath )
