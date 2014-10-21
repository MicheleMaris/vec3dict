import numpy as _np
class vec3dict :
   """ handles a 3d vector both as a dictionary and an indexed array """
   def __init__(self,*karg) :
      """ vec3dict() : empty vector 3d (Values are None)
          vec3dict(0) : vector 3d with all the elements [int(0), int(0), int(0)]
          vec3dict(5) : vector 3d with all the elements int(5)
          vec3dict(1,3,5) : vector 3d with all elements [int(1), int(3), int(5)]
          vec3dict('x') : vector 3d [1.,0.,0.]
      """
      import numpy as np
      self.v=[None,None,None]
      if len(karg) == 0 :
         return 
      if type(karg[0]) == type('') :
         self.v=[0.,0.,0.]
         if karg[0] == 'x' or karg[0] == 'X' :
            self.v[0]=1.
         elif karg[0] == 'y' or karg[0] == 'Y' :
            self.v[1]=1.
         elif karg[0] == 'z' or karg[0] == 'Z' :
            self.v[2]=1.
         else :
            raise Exception('invalid versor name')
         self.v=np.array(self.v)
         return
      if type(karg[0]) == type(vec3dict()) :
         self.v[0] = karg[0][0]
         self.v[1] = karg[0][1]
         self.v[2] = karg[0][2]
         self.v=np.array(self.v)
         return
      if len(karg) < 3 :
         for k in range(3) :
            self.v[k]=karg[0]
         self.v=np.array(self.v)
      else :
         self.v[0]=karg[0]
         self.v[1]=karg[1]
         self.v[2]=karg[2]
         self.v=np.array(self.v)
   def name2index(self,key):
      "return the index for a given component name"
      if key=='x' :
         return 0
      elif key=='y' :
         return 1
      elif key=='z' :
         return 2
      else :
         raise Exception('invalid keyword')
   def index2name(self,idx):
      "return the name for a given component index"
      try :
         return ['x','y','z'][idx]
      except :
         raise Exception('invalid index')
   def ismybrother(self,that) :
      return that.__class__.__name__ == self.__class__.__name__ 
   def __len__(self) :
      try :
         return self.v.shape[1]
      except :
         return 1
   def linterp(self,t,left=-_np.inf,right=_np.inf,noTest=False) :
      import numpy as np
      from scipy import interpolate
      if noTest :
         it=np.array(np.floor(t),dtype='int')
         dt=t-it
         x=(self['x'][it+1]-self['x'][it])*dt+self['x'][it]
         y=(self['y'][it+1]-self['y'][it])*dt+self['y'][it]
         z=(self['z'][it+1]-self['z'][it])*dt+self['z'][it]
         return vec3dict(x,y,z)
      x=np.zeros(t.shape)
      y=np.zeros(t.shape)
      z=np.zeros(t.shape)
      it=np.array(np.floor(t),dtype='int')
      idx=np.where(it < 0)[0]
      if len(idx) > 0 :
         x[idx]=left
         y[idx]=left
         z[idx]=left
      idx=np.where(it > len(self)-1)[0]
      if len(idx) > 0 :
         x[idx]=right
         y[idx]=right
         z[idx]=right
      idx=np.where(it == len(self)-1)[0]
      if len(idx) > 0 :
         x[idx]=self['x'][-1]
         y[idx]=self['x'][-1]
         z[idx]=self['x'][-1]
      idx=np.where((0<=it)*(it<len(self)-1))[0]
      if len(idx) > 0 :
         dt=t[idx]-it[idx]
         it=it[idx]
         x[idx]=(self['x'][it+1]-self['x'][it])*dt+self['x'][it]
         y[idx]=(self['y'][it+1]-self['y'][it])*dt+self['y'][it]
         z[idx]=(self['z'][it+1]-self['z'][it])*dt+self['z'][it]
      return vec3dict(x,y,z)
   def argslice(self,idx) :
      out=vec3dict()
      out['x']=self['x'][idx]
      out['y']=self['y'][idx]
      out['z']=self['z'][idx]
      return out
   def __getitem__(self,idx) :
      import numpy as np
      if (idx.__class__ == np.zeros(2).__class__) :
         return self.argslice(idx)
      try :
         idx = int(idx)
         try :
            return self.v[idx]
         except :
            raise Exception('out of bounds')
      except :
         if idx=='x' :
            i=0
         elif idx=='y' :
            i=1
         elif idx=='z' :
            i=2
         else :
            raise Exception('invalid keyword')
         return self.v[i]
   def __setitem__(self,idx,that) :
      try :
         idx = int(idx)
         try :
            self.v[idx] = that
         except :
            raise Exception('out of bounds')
      except :
         if idx=='x' :
            i=0
         elif idx=='y' :
            i=1
         elif idx=='z' :
            i=2
         else :
            raise Exception('invalid keyword')
         self.v[i] = that
   def __add__(self,that) :
      new = vec3dict(self)
      if self.ismybrother(that):
         for k in range(3) :
            new.v[k]+=that[k]
      else :
         for k in range(3) :
            new.v[k]+=that
      return new
   def __sub__(self,that) :
      new = vec3dict(self)
      if self.ismybrother(that) :
         for k in range(3) :
            new.v[k]-=that[k]
      else :
         for k in range(3) :
            new.v[k]-=that
      return new
   def __div__(self,that) :
      new = vec3dict(self)
      if not self.ismybrother(that) :
         for k in range(3) :
            new.v[k]=new.v[k]/that
      else :
         raise Exception('right side not a scalar')
      return new
   def __mul__(self,that) :
      if not self.ismybrother(that) :
         new = vec3dict(self)
         for k in range(3) :
            new.v[k]*=that
         return new
      else :
         raise Exception('right side not a scalar')
   def __rmul__(self,that) :
      if not self.ismybrother(that) :
         new = vec3dict(self)
         for k in range(3) :
            new.v[k]*=that
         return new
      else :
         raise Exception('left side not a scalar')
   def __neg__(self) :
      return vec3dict(-self.v[0],-self.v[1],-self.v[2])
   def __str__(self) :
      return 'x : '+str(self.v[0])+', y : '+str(self.v[1])+', z : '+str(self.v[2])
   def mul_by_array(self,that) :
      """
         multiplies by an array, component by component
         it is used as an example to generate a multicomponent vec3dict
         Example:
            arange(10)*vec3dict(1,0,0) 
         will result in a list of vec3dict objects with one component, while
            vec3dict(1,0,0).by_array(arange(10)) 
         will result in a single vec3dict objects with 10 elements in each column
         BeWare: the resulting Vec3Dict is not granted to work with all the methods
      """
      return vec3dict(self.v[0]*that,self.v[1]*that,self.v[2]*that)
   def add_array(self,that) :
      """
         add an array, component by component
         it is used as an example to generate a multicomponent vec3dict
         Example:
            arange(10)+vec3dict(1,0,0) 
         will result in an error
            vec3dict(1,0,0).add_array(arange(10)) 
         will result in a single vec3dict objects with 10 elements in each column
         which are the sum of 
            vec3dict(1,0,0)['x'], vec3dict(1,0,0)['y'], vec3dict(1,0,0)['z']
         with the array
      """
      return vec3dict(self.v[0]+that,self.v[1]+that,self.v[2]+that)
   def dot(self,that) :
      """ dot product """
      if not self.ismybrother(that) :
         raise Exception('right side not a vector')
      else :
         return self.v[0]*that.v[0]+self.v[1]*that.v[1]+self.v[2]*that.v[2]
   def norm(self) :
      """ returns the norm """
      return (self.v[0]*self.v[0]+self.v[1]*self.v[1]+self.v[2]*self.v[2])**0.5
   def keys(self) :
      """ return names of elements """
      return ['x','y','z']
   def array(self,dtype=None) :
      """ returns an array """
      from numpy import array
      if dtype == None :
         return array(self.v)
      return array(self.v,dtype=dtype)
   def dict(self) :
      """ returns a dictionary """
      return {'x':self.v[0],'y':self.v[1],'z':self.v[2]}
   def ext(self,that) :
      if not self.ismybrother(that) :
         raise Exception('right side not a vector')
      else :
         new = vec3dict(0)
         new[0]=self.v[1]*that.v[2]-self.v[2]*that.v[1]
         new[1]=-self.v[0]*that.v[2]+self.v[2]*that.v[0]
         new[2]=self.v[0]*that.v[1]-self.v[1]*that.v[0]
         return new
   def norm_ext(self,that) :
      "norm of an external product"
      if not self.ismybrother(that) :
         raise Exception('right side not a vector')
      else :
         new=(self.v[1]*that.v[2]-self.v[2]*that.v[1])**2
         new+=(self.v[0]*that.v[2]-self.v[2]*that.v[0])**2
         new+=(self.v[1]*that.v[0]-self.v[0]*that.v[1])**2
         return new**0.5
   def angle(self,that) :
      "angle between two vectors, radiants"
      import numpy as np
      if not self.ismybrother(that) :
         raise Exception('right side not a vector')
      else :
         Dot=self.dot(that)
         Ext=self.norm_ext(that)
         return np.arctan2(Ext,Dot)
   def copy(self) :
      import copy
      return copy.deepcopy(self)
   def versor(self) :
      "returns the versor"
      new=self.copy()
      nn=new.norm()
      for k in new.keys() : new[k]=new[k]/nn
      return new
   def mean(self) :
      "returns the averaged vector"
      return vec3dict(self.v[0].mean(),self.v[1].mean(),self.v[2].mean())
         
#class matr3dict :
   #""" handles a 3x3 matrix vector both as a dictionary and an indexed array , note rows are 3d matrices"""

if __name__=='__main__' :
   import numpy as np
   v1=vec3dict()
   print v1
   
   v2=vec3dict(1)
   print v2

   v3=vec3dict(1,2,3)
   print v3
   
   v4=vec3dict(v3)
   print v4
   
   print -v4
   
   print 2.*v4
   print v4+3.
   print v4.norm()
   print v4 - v4
   print v4.dot(v4)
   print v4.ext(v4+vec3dict(0.,0.,1.))
   print v4+vec3dict(0.,0.,1.)
   
   vx=vec3dict(1.,0.,0.)
   vy=vec3dict(0.,1.,0.)
   vz=vec3dict(0.,0.,1.)
   
   print
   print "||VX x VY|| ",vx.norm_ext(vy)
   print "||VX x V45degXY|| ",vx.norm_ext(vec3dict(np.cos(np.pi/4),np.cos(np.pi/4),0.))
   print "||VX x V45degXZ|| ",vx.norm_ext(vec3dict(np.cos(np.pi/4),0.,np.cos(np.pi/4)))
   print
   print "angle(vx,vy) = ",vx.angle(vy)*180./np.pi
   print "angle(vx,vz) = ",vx.angle(vz)*180./np.pi
   print "angle(vy,vz) = ",vy.angle(vz)*180./np.pi
   print
   print "angle(vx,V45degXY) = ",vx.angle(vec3dict(np.cos(np.pi/4),np.cos(np.pi/4),0.))*180./np.pi
   print "angle(vx,V45degXZ) = ",vx.angle(vec3dict(np.cos(np.pi/4),0.,np.cos(np.pi/4)))*180./np.pi
   
