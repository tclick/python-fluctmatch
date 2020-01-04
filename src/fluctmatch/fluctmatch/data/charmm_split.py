# ------------------------------------------------------------------------------
#   python-fluctmatch
#   Copyright (c) 2013-2020 Timothy H. Click, Ph.D.
#
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#   Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#   Neither the name of the author nor the names of its contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS”
#    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#    ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
#    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
#    OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
#    DAMAGE.
#
#   Timothy H. Click, Nixon Raj, and Jhih-Wei Chu.
#   Simulation. Meth Enzymology. 578 (2016), 327-342,
#   Calculation of Enzyme Fluctuograms from All-Atom Molecular Dynamics
#   doi:10.1016/bs.mie.2016.05.024.
#
# ------------------------------------------------------------------------------

split_inp = ("""
    * Create a subtrajectory from a larger CHARMM trajectory.
    * This is for <= c35.
    *

    set version {version}
    if @version .ge. 36 then
        dimension chsize 500000 maxres 3000000
    endif
    bomlev -5 ! This is for CHARMM 39

    set toppar {toppar}
    set begin {start}
    set end {stop}

    if @version .lt. 36 then
        read rtf  card name @toppar/top_all27_prot_na.rtf
        read para card {flex} name @toppar/par_all27_prot_na.prm

        read rtf  card name @toppar/top_all27_lipid.rtf append
        read para card {flex} name @toppar/par_all27_lipid.prm append

        read rtf  card name @toppar/top_all35_sugar.rtf append
        read para card {flex} name @toppar/par_all35_sugar.prm append

        stream @toppar/stream/toppar_water_ions.str
    else
        read rtf  card name @toppar/top_all36_prot.rtf
        read para card {flex} name @toppar/par_all36_prot.prm

        read rtf  card name @toppar/top_all36_na.rtf append
        read para card {flex} name @toppar/par_all36_na.prm append

        read rtf  card name @toppar/top_all36_carb.rtf append
        read para card {flex} name @toppar/par_all36_carb.prm append

        read rtf  card name @toppar/top_all36_lipid.rtf append
        read para card {flex} name @toppar/par_all36_lipid.prm append

        read rtf  card name @toppar/top_all36_cgenff.rtf append
        read para card {flex} name @toppar/par_all36_cgenff.prm append

        stream @toppar/toppar_water_ions.str
    endif

    read psf card name {psf}

    set iu    20
    set ou    30

    ! Load the trajectory
    open read  unit @iu file name {trajectory}
    open write unit @ou file name {outfile}

    ! Gather information from the first trajectory assuming that all trajectories
    ! are similar.
    traj query unit @iu

    if @start .gt. 1 then
        calc nother @start - 1
    else
        set nother @start
    endif
    traj first @iu skip ?SKIP begin @start stop @stop -
        iwrite @ou noth @nother

    close @iu
    close @ou
    stop
    """)
