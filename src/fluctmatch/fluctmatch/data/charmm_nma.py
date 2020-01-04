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

nma = """
    * Normal mode analysis of structure for parameter fitting. The original CHARMM
    * script was written by Prof. Jhih-Wei Chu
    *

    {dimension}

    set version {version}
    bomlev -5 ! This is for CHARMM 39

    ! Additional information
    set temp    {temperature}
    set fileu   10
    set fluctu  20
    set vibu    30

    ! Open CHARMM topology and parameter file
    read rtf  card name "{topology_file}"
    read para card {flex} name "{fixed_prm}"

    ! Open PSF and coordinate files
    if @version .ge. 39 then
        read psf  card name "{xplor_psf_file}"
    else
        read psf  card name "{psf_file}"
    endif
    read coor card name "{crd_file}"
    coor copy comp

    skip all excl bond
    update inbfrq 0

    ener

    ! Minimize structure using steepest descent and ABNR
    mini   sd nstep 100
    mini abnr nstep 2000

    coor orie rms mass
    scalar wmain copy mass

    ioformat extended
    write coor card name "{nma_crd}"

    stream "{stream_file}"

    ic fill
    write ic unit @fileu card resid name "{avg_ic}"

    calc nmode   ?natom * 3

    set nmodes   @nmode
    set type     temp

    open write unit @fluctu card name "{fluct_ic}"
    open write unit @vibu   card name "{nma_vib}"

    ! Perform normal mode analysis at desired temperature for vibrational normal
    ! modes
    vibran nmode @nmodes
        diag fini
        fluc ic @type @temp tfre 0.0 mode 7 thru @nmodes
        ic save
        ic write unit @fluctu resid
        write normal card mode 1 thru @nmodes unit @vibu
    end

    stop
    """
