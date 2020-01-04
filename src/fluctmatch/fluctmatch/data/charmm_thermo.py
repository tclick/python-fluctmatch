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

thermodynamics = """
    * Calculate thermodynamic qunatities for an ENM structure.
    *

    {dimension}

    set version {version}
    bomlev -5 ! This is for CHARMM 39

    ! Additional data
    set ndcd    1
    set temp    {temperature}
    set fileu   10

    ! Open CHARMM topology and parameter file
    read rtf  card name "{topology_file}"
    read para card {flex} name "{fixed_prm}"

    ! Open PSF and coordinate files
    if @version .ge. 39 then
        read psf card name "{xplor_psf_file}"
    else
        read psf card name "{psf_file}"
    endif
    read coor card name "{crd_file}"
    coor copy comp

    stream "{stream_file}"

    skip all excl bond
    update inbfrq 0

    ! Load the trajectory
    open read  unit @fileu file name {trajectory}

    ! Gather information from the first trajectory assuming that all trajectories
    ! are similar.
    traj query unit @fileu

    calc nmod = 3*?NATOM
    vibran nmodes @nmod
        coor dyna sele all end nopr first @fileu nunit @ndcd begin ?START -
            nskip ?SKIP orient sele all end
        quasi first @fileu nunit @ndcd nskip ?SKIP begin ?START sele all end -
            temp @temp thermo resi
    end
    calc ts = ?stot * @temp
    close unit @fileu

    stop
    """
