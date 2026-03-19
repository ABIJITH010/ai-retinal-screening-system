import api from "./api";

export interface RegisterRequest {
  name: string;
  email: string;
  password: string;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface UserResponse {
  id: number;
  name: string;
  email: string;
}

export async function registerUser(request: RegisterRequest) {
  const { data } = await api.post<UserResponse>("/auth/register", request);
  return data;
}

export async function loginUser(request: LoginRequest) {
  const { data } = await api.post<TokenResponse>("/auth/login", request);
  return data;
}

export async function getMe() {
  const { data } = await api.get<UserResponse>("/auth/me");
  return data;
}

