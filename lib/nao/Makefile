
include arch.inc
sinclude libs.inc

TARGET=../libnao.so

OBJS = \
  m_io.o \
  m_null2char.o \
  m_color.o \
  m_die.o \
  m_log.o \
  m_warn.o \
  m_precision.o \
  m_siesta_wfsx.o \
  m_siesta_hsx.o \
  m_fact.o   \
  m_rsphar.o \
  cputime.o

ifndef LINK
$(error "LINK is not set: define in arch.inc file ?")
endif

all: $(OBJS)
	$(LINK) $(LFLAGS) --shared $(OBJS) $(USER_LIBS) -o $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) *.mod


%.o:%.F90 arch.inc
	$(TIME) $(FORTRAN) $(FLAGS) $(INCDIR) -c $<

%.o:../%.c arch.inc
	$(TIME) $(CC) $(CFLAGS) $(INCDIR) -c $<

