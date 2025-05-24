import { SetMetadata } from '@nestjs/common';

export enum UserRole {
  TEACHER = 'TEACHER',
  STUDENT = 'STUDENT',
}

export const Roles = (...roles: UserRole[]) => SetMetadata('roles', roles); 