!
! This file is part of SusHi
!
C----------------------------------------------------------------------
c..
c..   readslha.f
c..
c..   needs the SUBROUTINE SLHABLOCKS() in slhablocks.f.
c..
c..   NOTE: maximum length of input lines is currently 
c..         set to 200 characters.
c..
c..   To add a block:
c..   (1) add an array for its value in common-slha.f: NEWARRAY(100)
c..   (2) possibly add an array for the names of its entries in 
c..       common-slha.f:  CNEWARRAY(100)
c..   (3) add a section in SUBROUTINE SLHABLOCKS() that fills NEWARRAY.
c..
C----------------------------------------------------------------------

C-{{{ subroutine readblocks:

      subroutine readblocks(iunit,blocks,nkeys)
c..
c..   This is just an interface to readslha to read multiple blocks.
c..   
c..   Example: 
c..   BLOCKS = ('mass','sminputs','minpar')
c..
      implicit none
      integer i,ifound,iunit
      integer nkeys(*)
      character(*) blocks(*)
      character*15 block

      i=1
      do while (blocks(i).ne.' ')
         call readslha(iunit,blocks(i),ifound)
         nkeys(i) = ifound
         if (ifound.eq.0) then
            write(6,*) 'No Block ',blocks(i),' found.'
         endif
         i=i+1
      enddo

      end

C-}}}
C-{{{ subroutine readslha:

      subroutine readslha(iunit,blocktypin,ifound)
c..
c..   Read block BLOCKTYPIN from SLHA input file
c..   and fill the corresponding COMMON block.
c..   SLHA input file must be open in unit IUNIT.
c..
c..   Example:
c..   BLOCKTYPIN = 'mass'
c..   
c..   If BLOCK is found in input file, IFOUND is set to 1.
c..   IFOUND = 0 otherwise.
c..   
      implicit none
      integer i,j,ierr,iread,lcount,lenblock,klen,ifound,iunit
      character cline*200,clineout*200,emptyline*200
      character blocktype*15,blocktypin*15
      character uppercase*1

      blocktype = '               '
      blocktype = blocktypin
      
      rewind(iunit)

      emptyline = ''
      do i=1,20
         emptyline = emptyline//'          '
      enddo
      cline = emptyline
      lenblock = lnblnk(blocktype)
      
c..   change input to upper case:
      do j=1,lenblock
         blocktype(j:j)=uppercase(blocktype(j:j))
      enddo
      lcount=0
      iread=0
      ifound=0
c..   I use a GOTO loop, because I don't know any other way
c..   how to read to the end of a file without knowing its length.
c..   Henry's edits; prompt the python program to give input
      write(*,*) '**send input file to stdin'
 100  lcount = lcount+1
c..   Henry's edits; instead of readin from file, read from stdin
      read(*,1001,END=101) cline
c..   Henry's edits; tricky to send EOF down pipe
c..   Henry's edits; this is a substitute
      if(INDEX(cline, "END").GE.1) then
          write(*, *) '**detected END'
          GOTO 101
      endif

      call normalline(cline,clineout)
c..   
c..   cline:    original input with comments and multiple blanks removed
c..   clineout: uppercase of cline
c..   
      cline = clineout
      do i=1,200
         clineout(i:i) = uppercase(clineout(i:i))
      enddo
c..   check if line is valid input line:
      if ((clineout(1:1).ne.' ').and.(clineout(1:1).ne.'#').and.
     &     (clineout(1:1).ne.'B').and.(clineout(1:1).ne.'D')) then
         write(6,*) '*** Error reading SLHA input file (UNIT=',iunit
     &        ,'):'
         write(6,*) 'Line ',lcount,': ',
     &        clineout(1:1),' = char(',
     &        ichar(clineout(1:1)),') not allowed in first column'
         stop
      endif
      if (clineout.eq.emptyline) goto 100
c..   If we are looking at a keyword (BLOCK, DECAY), then either
c..   - start reading data if it's the correct BLOCK
c..   - stop reading if we are already reading data
      if ((clineout(1:5).eq.'BLOCK').or.(clineout(1:5).eq.'DECAY')) then
         klen = 7
         do while (clineout(klen:klen).ne.' ') 
            klen=klen+1
         enddo
         klen=klen-1
         iread = 0
         if ((klen.eq.lenblock+6).and.
     &        (clineout(1:klen).eq.'BLOCK '//blocktype(1:lenblock)))
     &        then
            ifound = 1
            iread = 1
            goto 100
         endif
      endif
      if (iread.eq.1) then
         call slhablocks(blocktype,cline,ierr)
         if (ierr.eq.1) then
            write(6,*) '*** Error reading SLHA input file (UNIT
     &           =',iunit,'):'
            write(6,*) 'Line ',lcount,':>',cline(1:60)
            stop
         endif
      endif
      goto 100
 1001 format(200a)
 101  continue
      end

C-}}}
C-{{{ subroutine normalline:

      subroutine normalline(clinein,clineout)
c..
c..   Remove multiple blanks, remove comments,
c..   and change everything to upper case
c..
c..   Example:
c..   clinein = '  some    string with   tabs and  #  comments'
c..   
c..   Returns:
c..   clineout = ' some string with tabs and     '
c..   
c..   The output string is always 200 chars long (trailing blanks)
c..   
      implicit none
      integer i,j
      character cline*200,clinein*200,clineinv*200,clineout*200
      logical nonempty

      cline = clinein
c..   remove comments, change tabs to spaces, and change to upper case:
      do i=1,200
         if (cline(i:i).eq.char(9)) cline(i:i) = ' '
         if (cline(i:i).eq.char(13)) cline(i:i) = ' '
         if (cline(i:i).eq.'#') then
            do j=i,200
               cline(j:j) = ' '
            enddo
         endif
      enddo

c..   remove multiple whitespace:
      do i=1,200
         clineinv(i:i) = cline(200+1-i:200+1-i)
      enddo
      clineout=' '
      i=1
      nonempty = .false.
      do while (i.le.200)
!     this fixes the out of bounds runtime error
         if(clineinv(i:i).ne.' ') then
            do while(i.le.200)
               if(clineinv(i:i).ne.' ') then
                  nonempty = .true.
                  clineout = clineinv(i:i)//clineout
                  i = i + 1
               else
                 exit
               endif
            enddo
            if (i.le.200) then
               clineout = ' '//clineout(1:199)
               i = i + 1
            endif
         else
            do while(i.le.200)
               if(clineinv(i:i).eq.' ') then
                  i = i + 1
               else
                  exit
               endif
            enddo
         endif

!     the code below causes an out of bounds runtime error
!     because the argument of the do while could get i = 201
c        do while ((i.le.200).and.(clineinv(i:i).ne.' '))
c           nonempty = .true.
c           clineout = clineinv(i:i)//clineout
c           i=i+1
c        enddo
c        if (i.le.200) then
c           clineout = ' '//clineout(1:199)
c           i=i+1
c        endif
c        do while ((i.le.200).and.(clineinv(i:i).eq.' '))
c           i=i+1
c        enddo
      enddo
      end

C-}}}
C-{{{ function uppercase:

      function uppercase(str)

      implicit none
      integer ich
      character*1 str,uppercase

      do ich=97,122
         if (char(ich).eq.str) then
            uppercase=char(ich-32)
            return
         else
            uppercase=str
         endif
      enddo
      
      end

C-}}}
