This is the NEKbone codebase with implementation of compute communication overlap approach.
This codebase is used for Pathforward MS_1-6 submission.

User can select the mode before compilation by modifying the relevant PPLIST in the /test/example*/makenek-* files. Three modes which are available in this codebase are:
1. BASELINE: Compilation using PPLIST="TIMERS CGTIMERS"                                                	#PPLIST for nekbone baseline mode
2. OVERLAP_MOD: Compilation using PPLIST="TIMERS CGTIMERS HPE_OVERLAP_MOD"				#PPLIST for nekbone code with partial overlap mode	
3. OVERLAP_ALL: Compilation using PPLIST="TIMERS CGTIMERS HPE_OVERLAP_MOD HPE_OVERLAP_ALL"		#PPLIST for nekbone code with complete overlap mode

#Compilation using intel-tools
- Go to the test/example1_Intel directory
- Set up the environment for intel compiler and intel MPI libraries.
- Edit the SIZE file with appropriate values for "lelt" and "np"
- Edit the makenek-intel file to select the relevant PPLIST depending upon the mode you want to compile with.
- Run "./makenek-intel" command.

#Run
- Edit the data.rea file with the NEKbone problem size of interest.
- Customize the sample PBS script and submit the runs on cluster with PBS scheduler.
